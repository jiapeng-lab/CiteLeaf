"""Keyword (BM25) recall, reciprocal-rank fusion, and cross-encoder reranking.

These three stages sit beside the vector store so the two recall paths and the
fusion/rerank steps can be unit-tested in isolation. jieba and the
cross-encoder are imported lazily: importing this module never triggers a
model download or a jieba dictionary build.
"""

import json
import logging
import math
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .documents import Chunk

logger = logging.getLogger("citeleaf.retrieval")

# RRF 的平滑常数，按题目固定为 60。
RRF_K = 60
# BM25Okapi 的默认超参（与 rank_bm25 一致）。
_BM25_K1 = 1.5
_BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """用 jieba 对中文分词，返回去除空白后的词条列表。

    首次调用会构建 jieba 前缀词典（约 1 秒），并把“Building prefix dict”
    这类提示降到 WARNING，避免污染界面输出。
    """
    import jieba

    if not getattr(tokenize, "_quieted", False):
        jieba.setLogLevel(logging.WARNING)
        tokenize._quieted = True
    return [token for token in jieba.lcut(text) if token.strip()]


def rrf_fuse(ranked_lists: Iterable[list[Chunk]], k: int = RRF_K) -> list[Chunk]:
    """用 Reciprocal Rank Fusion 合并多路排序结果。

    每个片段在每一路中的得分是 ``1 / (k + rank)``，rank 从 1 起（榜首项得
    ``1 / (k + 1)``）；同一片段出现在多路时分数累加。分数相同时保持「首次
    出现」顺序（stable sort），因此并列名次可确定复现。
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    order: list[str] = []
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, 1):
            key = chunk.id
            if key not in scores:
                scores[key] = 0.0
                chunks[key] = chunk
                order.append(key)
            scores[key] += 1.0 / (k + rank)
    order.sort(key=lambda key: scores[key], reverse=True)
    return [chunks[key] for key in order]


class BM25Index:
    """基于 jieba 分词的 BM25Okapi 倒排索引，随知识库持久化到 JSON。

    索引保存 ``doc_freqs``（词条 → 文档频次）和每个片段的词频表；加载时
    只需重新计算 IDF，既不对原文重新分词，也不在每次查询时重建。查询时的
    IDF 使用全库文档频次（与向量检索「全库索引 + 范围过滤」的做法一致）。
    """

    def __init__(self, k1: float = _BM25_K1, b: float = _BM25_B, tokenizer=None):
        self.k1, self.b = k1, b
        self.tokenizer = tokenizer or tokenize
        self.docs: list[dict] = []
        self.doc_freqs: dict[str, int] = {}
        self._rows: dict[str, list[int]] = {}  # doc_id -> 行下标
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0

    # ---- 构建 ----
    def add(self, chunk: Chunk) -> None:
        tf = Counter(self.tokenizer(chunk.text))
        row = {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "filename": chunk.filename,
            "page": chunk.page,
            "index": chunk.index,
            "text": chunk.text,
            "tf": dict(tf),
            "doc_len": sum(tf.values()),
        }
        index = len(self.docs)
        self.docs.append(row)
        self._rows.setdefault(chunk.doc_id, []).append(index)
        for term in tf:
            self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        self._recompute_idf()

    def add_documents(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            self.add(chunk)

    def remove(self, doc_id: str) -> None:
        """删除一个文档的全部片段；删除较罕见，整体重建词频表更不易出错。"""
        if doc_id not in self._rows:
            return
        self.docs = [row for row in self.docs if row["doc_id"] != doc_id]
        self._rebuild_lookup()

    def _rebuild_lookup(self) -> None:
        self.doc_freqs = {}
        self._rows = {}
        for index, row in enumerate(self.docs):
            self._rows.setdefault(row["doc_id"], []).append(index)
            for term in row["tf"]:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        self._recompute_idf()

    def _recompute_idf(self) -> None:
        """Lucene 风格的非负 IDF：``ln(1 + (N - df + 0.5)/(df + 0.5))``。

        相比经典 BM25Okapi 的 ``ln((N - df + 0.5)/(df + 0.5))``，加 1 保证
        高频词不会得到负权重，也避免小语料下「全部负分 → 被过滤」的边界。
        """
        n = len(self.docs)
        if not n:
            self._idf, self._avgdl = {}, 0.0
            return
        self._avgdl = sum(row["doc_len"] for row in self.docs) / n
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freqs.items()
        }

    # ---- 检索 ----
    def search(self, query: str, doc_ids: list[str], k: int = 6) -> list[Chunk]:
        """在指定文档范围内做关键词检索，返回按 BM25 分数降序的片段。"""
        if not doc_ids or not self.docs:
            return []
        qterms = Counter(self.tokenizer(query))
        scope = set(doc_ids)
        rows = [i for did in scope for i in self._rows.get(did, [])]
        avgdl = self._avgdl or 1.0
        scored: list[tuple[float, dict]] = []
        for i in rows:
            row = self.docs[i]
            score = 0.0
            tf = row["tf"]
            dl = row["doc_len"]
            for term in qterms:
                idf = self._idf.get(term)
                if not idf:
                    continue
                freq = tf.get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * dl / avgdl)
                score += idf * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [self._to_chunk(row) for _, row in scored[:k]]

    @staticmethod
    def _to_chunk(row: dict) -> Chunk:
        return Chunk(
            id=row["id"],
            doc_id=row["doc_id"],
            filename=row["filename"],
            page=row["page"],
            index=row["index"],
            text=row["text"],
        )

    # ---- 持久化 ----
    def save(self, path: Path) -> None:
        data = {"version": 1, "k1": self.k1, "b": self.b,
                "docs": self.docs, "doc_freqs": self.doc_freqs}
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path, tokenizer=None) -> "BM25Index":
        if not path.exists():
            return cls(tokenizer=tokenizer)
        data = json.loads(path.read_text(encoding="utf-8"))
        index = cls(k1=data.get("k1", _BM25_K1), b=data.get("b", _BM25_B),
                    tokenizer=tokenizer)
        index.docs = data["docs"]
        index.doc_freqs = dict(data["doc_freqs"])
        index._rows = {}
        for i, row in enumerate(index.docs):
            index._rows.setdefault(row["doc_id"], []).append(i)
        index._recompute_idf()
        return index


class Reranker:
    """跨编码器精排：懒加载、走 CPU；失败时回退到输入顺序而非阻断问答。"""

    def __init__(self, model_name: str, model=None):
        self.model_name = model_name
        self._model = model  # 可注入替身，测试时避免触发真实下载

    def _load(self):
        if self._model is None:
            # 本地目录直接加载；仅 Hugging Face 模型标识才提示下载。
            if Path(self.model_name).exists():
                logger.info("加载本地重排模型 %s", self.model_name)
            else:
                logger.warning(
                    "首次启用重排将下载重排模型 %s（体积较大，请保持网络畅通）。",
                    self.model_name,
                )
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        if not chunks:
            return []
        try:
            model = self._load()
            scores = model.predict([(query, chunk.text) for chunk in chunks])
            # 分数相同时按片段在原文档中的顺序稳定排序，保证结果可复现。
            ranked = sorted(
                zip(scores, chunks, strict=False),
                key=lambda pair: (-float(pair[0]), pair[1].index),
            )
            return [chunk for _, chunk in ranked[:top_k]]
        except Exception as exc:  # 下载失败/模型不可用时宁可回退也不阻断问答
            logger.warning("重排失败，回退到精排前的候选顺序：%s", exc)
            return chunks[:top_k]
