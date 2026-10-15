"""Local semantic index. Explicit vectors prevent hidden model downloads by Chroma."""

import hashlib
from pathlib import Path
from threading import RLock

from .documents import Chunk, parse_file


class BGEEncoder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device="cpu")

    def encode(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        if query:
            texts = ["为这个句子生成表示以用于检索相关文章：" + text for text in texts]
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        ).tolist()


class Knowledge:
    def __init__(self, directory: Path, encoder, model_id: str = "BAAI/bge-small-zh-v1.5"):
        import chromadb

        self.encoder, self.lock = encoder, RLock()
        self.client = chromadb.PersistentClient(path=str(directory))
        # Different vector models/chunking versions must never share an index.
        fingerprint = hashlib.sha256((model_id + ":420:60:v1").encode()).hexdigest()[:12]
        self.collection = self.client.get_or_create_collection(
            name=f"citeleaf_{fingerprint}",
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
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
            vectors = self.encoder.encode([c.text for c in chunks])
            try:
                for offset in range(0, len(chunks), 128):
                    batch = chunks[offset : offset + 128]
                    self.collection.add(
                        ids=[c.id for c in batch],
                        documents=[c.text for c in batch],
                        metadatas=[c.metadata() for c in batch],
                        embeddings=vectors[offset : offset + 128],
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
        data = self.collection.query(
            query_embeddings=self.encoder.encode([question], query=True),
            where=self.scope(doc_ids),
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        # Similarity ranks candidates; the generator separately judges direct support.
        return [
            Chunk(id=ident, text=text, **meta)
            for ident, text, meta in zip(
                data["ids"][0], data["documents"][0], data["metadatas"][0], strict=True
            )
        ]

    def delete(self, doc_id: str):
        with self.lock:
            self.collection.delete(where={"doc_id": doc_id})
