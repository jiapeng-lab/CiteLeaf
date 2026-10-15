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
    def __init__(self, model: str):
        from langchain_huggingface import HuggingFaceEmbeddings

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
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)


class Knowledge:
    def __init__(self, directory: Path, encoder, model_id: str = "BAAI/bge-small-zh-v1.5"):
        import chromadb

        self.encoder, self.lock = encoder, RLock()
        self.client = chromadb.PersistentClient(path=str(directory))
        # Keep the persisted v1 schema/model namespace: existing vectors and sources
        # remain valid. The LangChain splitter uses the same 420/60 size limits.
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
        return {"doc_id": {"$in": doc_ids}}

    def ingest(self, content: bytes, filename: str) -> tuple[str, bool]:
        chunks = parse_file(content, filename)
        doc_id = chunks[0].doc_id
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
        data = self.collection.get(include=["metadatas"])
        counts = {}
        for meta in data["metadatas"]:
            row = counts.setdefault(meta["doc_id"], {**meta, "chunks": 0})
            row["chunks"] += 1
        return sorted(counts.values(), key=lambda x: x["filename"])

    def read(self, doc_ids: list[str]) -> list[Chunk]:
        if not doc_ids:
            return []
        data = self.collection.get(where=self.scope(doc_ids), include=["documents", "metadatas"])
        rows = [
            Chunk(id=ident, text=text, **meta)
            for ident, text, meta in zip(
                data["ids"], data["documents"], data["metadatas"], strict=True
            )
        ]
        return sorted(rows, key=lambda c: (c.filename, c.doc_id, c.index))

    def search(self, question: str, doc_ids: list[str], k: int = 6) -> list[Chunk]:
        if not doc_ids or not self.collection.count():
            return []
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
        with self.lock:
            self.collection.delete(where={"doc_id": doc_id})
