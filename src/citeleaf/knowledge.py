"""Local semantic index. Explicit vectors prevent hidden model downloads by Chroma."""

import hashlib
from pathlib import Path
from threading import RLock

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langsmith import tracing_context

from .documents import Chunk, parse_file


class BGEEncoder(Embeddings):
    """封装本地 BGE 模型，分别提供文档和查询的向量化接口。"""
    def __init__(self, model: str):
        from langchain_huggingface import HuggingFaceEmbeddings

        # 文档和查询使用相同的归一化向量；BGE 查询额外添加检索提示前缀。
        self.model = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={
                "normalize_embeddings": True,
                "prompt": "为这个句子生成表示以用于检索相关文章：",
            },
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """将多个文档片段转换为向量，用于建立索引。"""
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """将用户问题转换为查询向量，用于相似度检索。"""
        return self.model.embed_query(text)


class Knowledge:
    """管理本地向量知识库，提供文档入库、查询、读取和删除能力。"""
    def __init__(self, directory: Path, encoder, model_id: str = "BAAI/bge-small-zh-v1.5"):
        import chromadb

        self.encoder, self.lock = encoder, RLock()
        self.client = chromadb.PersistentClient(path=str(directory))
        # Keep the persisted v1 schema/model namespace: existing vectors and sources
        # remain valid. The LangChain splitter uses the same 420/60 size limits.
        # 模型或索引版本变化时使用不同集合，避免不同向量空间混用。
        fingerprint = hashlib.sha256((model_id + ":420:60:v1").encode()).hexdigest()[:12]
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=f"citeleaf_{fingerprint}",
            embedding_function=encoder,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.collection = self.client.get_collection(
            f"citeleaf_{fingerprint}", embedding_function=None
        )

    @staticmethod
    def scope(doc_ids: list[str]):
        """构造只匹配所选文档的 Chroma 过滤条件。"""
        return {"doc_id": {"$in": doc_ids}}

    def ingest(self, content: bytes, filename: str) -> tuple[str, bool]:
        """解析并索引文档；返回文档 ID 和是否实际新增。"""
        chunks = parse_file(content, filename)
        doc_id = chunks[0].doc_id
        # 去重和写入放在同一把锁内；批量写入失败时删除该文档的已写入片段。
        with self.lock:
            if self.collection.get(where={"doc_id": doc_id}, limit=1)["ids"]:
                return doc_id, False
            try:
                for offset in range(0, len(chunks), 128):
                    batch = chunks[offset : offset + 128]
                    self.vectorstore.add_documents(
                        documents=[
                            Document(page_content=c.text, metadata=c.metadata()) for c in batch
                        ],
                        ids=[c.id for c in batch],
                    )
            except Exception:
                self.collection.delete(where={"doc_id": doc_id})
                raise
        return doc_id, True

    def documents(self) -> list[dict]:
        """列出已入库文档及各自的片段数量。"""
        data = self.collection.get(include=["metadatas"])
        counts = {}
        for meta in data["metadatas"]:
            row = counts.setdefault(meta["doc_id"], {**meta, "chunks": 0})
            row["chunks"] += 1
        return sorted(counts.values(), key=lambda x: x["filename"])

    def read(self, doc_ids: list[str]) -> list[Chunk]:
        """按文档及原文顺序读取所选文档的全部片段，供摘要使用。"""
        if not doc_ids:
            return []
        data = self.collection.get(where=self.scope(doc_ids), include=["documents", "metadatas"])
        rows = [
            Chunk(id=ident, text=text, **meta)
            for ident, text, meta in zip(
                data["ids"], data["documents"], data["metadatas"], strict=True
            )
        ]
        # Chroma 返回顺序不保证是原文顺序，摘要前按文档和片段序号排序。
        return sorted(rows, key=lambda c: (c.filename, c.doc_id, c.index))

    def search(self, question: str, doc_ids: list[str], k: int = 6) -> list[Chunk]:
        """在所选文档中检索与问题语义相近的片段。"""
        if not doc_ids or not self.collection.count():
            return []
        # 过滤条件限制检索范围；相似度只负责排序，不代表答案一定有依据。
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": min(k, self.collection.count()), "filter": self.scope(doc_ids)},
        )
        # Similarity ranks candidates; the generator separately judges direct support.
        with tracing_context(enabled=False):
            documents = retriever.invoke(question)
        return [
            Chunk(
                id=doc.id or f"{doc.metadata['doc_id']}:{doc.metadata['index']}",
                text=doc.page_content,
                **doc.metadata,
            )
            for doc in documents
        ]

    def delete(self, doc_id: str):
        """删除指定文档的所有索引片段。"""
        with self.lock:
            self.collection.delete(where={"doc_id": doc_id})
