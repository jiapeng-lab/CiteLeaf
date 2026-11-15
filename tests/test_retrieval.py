"""Offline tests for hybrid retrieval: BM25, RRF fusion, and cross-encoder rerank."""

from langchain_core.embeddings import Embeddings

from citeleaf.documents import Chunk
from citeleaf.knowledge import Knowledge
from citeleaf.retrieval import BM25Index, Reranker, rrf_fuse, tokenize


def word_tokenizer(text):
    """测试替身：按空白切分，结果确定、无需词典。"""
    return text.split()


def chunk(doc_id, index, text):
    return Chunk(id=f"{doc_id}:{index}", doc_id=doc_id, filename=f"{doc_id}.txt",
                 page=1, index=index, text=text)


class FakeEncoder(Embeddings):
    """Test double only; production always uses a semantic BGE model."""

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


# ---- RRF ----

def test_rrf_fuses_two_lists_with_dedupe():
    a, b, c = chunk("d", 0, "a"), chunk("d", 1, "b"), chunk("d", 2, "c")
    # a 在两路都排第一，得分 1/61 + 1/61；b、c 各只有一路命中，得分 1/62。
    merged = rrf_fuse([[a, b], [a, c]])
    assert merged[0] is a
    assert [x.id for x in merged] == ["d:0", "d:1", "d:2"]  # b 先于 c：并列时保持首次出现顺序


def test_rrf_single_list_preserves_order():
    a, b = chunk("d", 0, "a"), chunk("d", 1, "b")
    assert rrf_fuse([[a, b]]) == [a, b]


def test_rrf_ignores_empty_lists():
    a, b = chunk("d", 0, "a"), chunk("d", 1, "b")
    assert rrf_fuse([[], [a, b]]) == [a, b]
    assert rrf_fuse([[], []]) == []


def test_rrf_ties_are_stable():
    a, b = chunk("d", 0, "a"), chunk("d", 1, "b")
    # 两路各自只有一条且不重叠：a、b 分数同为 1/61，保持首次出现顺序。
    merged = rrf_fuse([[a], [b]])
    assert [x.id for x in merged] == ["d:0", "d:1"]


# ---- BM25 ----

def test_bm25_scope_filter_and_ranking():
    idx = BM25Index(tokenizer=word_tokenizer)
    idx.add(chunk("a", 0, "年假 年假 申请"))
    idx.add(chunk("a", 1, "年假 申请"))
    idx.add(chunk("b", 0, "报销 发票"))
    hits = idx.search("年假 申请", ["a"], k=10)
    assert hits and all(c.doc_id == "a" for c in hits)
    # 词频更高、更集中的片段排前
    assert [c.id for c in hits] == ["a:0", "a:1"]
    # 范围过滤：只检索指定文档，范围外内容不出现
    assert [c.id for c in idx.search("报销", ["b"], k=10)] == ["b:0"]
    assert idx.search("年假", ["b"], k=10) == []
    assert idx.search("年假", [], k=10) == []


def test_bm25_persistence_roundtrip(tmp_path):
    path = tmp_path / "bm25.json"
    idx = BM25Index(tokenizer=word_tokenizer)
    idx.add(chunk("a", 0, "年假 年假 申请"))
    idx.add(chunk("b", 0, "报销 发票"))
    idx.save(path)
    loaded = BM25Index.load(path, tokenizer=word_tokenizer)
    assert [c.id for c in loaded.search("年假", ["a", "b"], 10)] == ["a:0"]
    assert loaded.search("报销", ["b"], 10)[0].doc_id == "b"


def test_bm25_remove_recomputes_frequencies():
    idx = BM25Index(tokenizer=word_tokenizer)
    idx.add(chunk("a", 0, "年假 申请"))
    idx.add(chunk("b", 0, "报销 发票"))
    idx.remove("a")
    assert idx.search("年假", ["a", "b"], 10) == []
    assert "年假" not in idx.doc_freqs


def test_bm25_load_missing_file(tmp_path):
    assert BM25Index.load(tmp_path / "nope.json", tokenizer=word_tokenizer).docs == []


def test_tokenize_chinese_splits_words():
    text = "员工年假共有十二个工作日"
    tokens = tokenize(text)
    assert tokens and all(t.strip() for t in tokens)
    # 断言多字词成词，而不是依赖 jieba 词典是否收录“年假”（0.42.1 未收录）。
    assert "员工" in tokens and "工作日" in tokens
    assert "".join(tokens) == text


# ---- Reranker ----

class FakeCrossEncoder:
    def __init__(self, scores=None, error=None):
        self.scores = scores or []
        self.error = error

    def predict(self, pairs):
        if self.error:
            raise self.error
        return self.scores


def test_rerank_reorders_by_score():
    chunks = [chunk("d", i, t) for i, t in enumerate(["甲", "乙", "丙"])]
    model = FakeCrossEncoder(scores=[0.1, 0.9, 0.5])
    out = Reranker("test", model=model).rerank("q", chunks, top_k=2)
    assert [c.id for c in out] == ["d:1", "d:2"]


def test_rerank_falls_back_to_input_order_on_error():
    chunks = [chunk("d", i, t) for i, t in enumerate(["甲", "乙", "丙"])]
    model = FakeCrossEncoder(error=RuntimeError("download failed"))
    out = Reranker("test", model=model).rerank("q", chunks, top_k=2)
    assert [c.id for c in out] == ["d:0", "d:1"]


# ---- Knowledge 级别的混合检索集成 ----

def test_hybrid_doc_id_filter_applies_to_keyword_path(tmp_path):
    kb = Knowledge(tmp_path / "index", FakeEncoder(), "test-double", hybrid_search=True)
    kb.reranker = FakeReranker(pick="all")
    a, _ = kb.ingest("年假为十二个工作日。".encode(), "a.txt")
    b, _ = kb.ingest("报销需要提交电子发票。".encode(), "b.txt")
    assert all(c.doc_id == a for c in kb.search("年假", [a]))
    # “年假”只出现在 a；范围限定 b 时，关键词路不得把 a 的片段混进来。
    assert all(c.doc_id == b for c in kb.search("年假", [b]))
    assert kb.search("年假", []) == []


class FakeReranker:
    def __init__(self, pick="first"):
        self.calls = []
        self.pick = pick

    def rerank(self, query, chunks, top_k):
        self.calls.append((query, list(chunks), top_k))
        return chunks[:1] if self.pick == "first" else chunks[:top_k]


def test_hybrid_rerank_stage_is_used(tmp_path):
    kb = Knowledge(tmp_path / "index", FakeEncoder(), "test-double", hybrid_search=True)
    kb.reranker = FakeReranker(pick="first")
    a, _ = kb.ingest("年假为十二个工作日。".encode(), "a.txt")
    b, _ = kb.ingest("报销需要提交电子发票。".encode(), "b.txt")
    hits = kb.search("年假", [a, b])
    assert len(hits) == 1  # 精排后按 top_k 收窄
    assert kb.reranker.calls and kb.reranker.calls[-1][0] == "年假"


def test_pure_vector_with_rerank_stage_is_used(tmp_path):
    kb = Knowledge(tmp_path / "index", FakeEncoder(), "test-double")
    assert not kb.hybrid_search  # 默认走单独 BGE 路线
    kb.reranker = FakeReranker(pick="first")
    a, _ = kb.ingest("年假为十二个工作日。".encode(), "a.txt")
    hits = kb.search("年假", [a])
    assert len(hits) == 1  # 纯向量 + 重排：先召回 top-20 再精排收窄
    assert kb.reranker.calls and kb.reranker.calls[-1][0] == "年假"


def test_bm25_backfills_missing_file_from_chroma(tmp_path):
    kb = Knowledge(tmp_path / "index", FakeEncoder(), "test-double")
    a, _ = kb.ingest("年假为十二个工作日。".encode(), "a.txt")
    for path in (tmp_path / "index").glob("bm25_*.json"):
        path.unlink()
    reopened = Knowledge(tmp_path / "index", FakeEncoder(), "test-double", hybrid_search=True)
    reopened.reranker = FakeReranker(pick="all")
    assert reopened.bm25.docs  # 已从 Chroma 回填
    assert any(c.doc_id == a for c in reopened.search("年假", [a]))


def test_delete_removes_bm25_rows(tmp_path):
    kb = Knowledge(tmp_path / "index", FakeEncoder(), "test-double", hybrid_search=True)
    a, _ = kb.ingest("年假为十二个工作日。".encode(), "a.txt")
    b, _ = kb.ingest("报销需要提交电子发票。".encode(), "b.txt")
    assert any(row["doc_id"] == a for row in kb.bm25.docs)
    kb.delete(a)
    assert all(row["doc_id"] != a for row in kb.bm25.docs)
    assert any(row["doc_id"] == b for row in kb.bm25.docs)
