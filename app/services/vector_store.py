"""
Vector store backed by ChromaDB (persisted to data/chroma_store — gitignored).
Upsert by chunk_id ensures re-ingesting a file replaces, not duplicates, chunks.
Cosine distance is converted to a [0, 1] similarity score before returning.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.utils.config import Settings
from app.services.chunker import CodeChunk
from app.utils.models import ChunkMeta, SourceChunk


class VectorStore:
    def __init__(self, settings: Settings):
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[
                {
                    "file_path": c.file_path,
                    "chunk_type": c.chunk_type,
                    "name": c.name or "",
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                }
                for c in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[SourceChunk]:
        n = min(top_k, self._collection.count() or 1)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append(SourceChunk(
                content=doc,
                metadata=ChunkMeta(
                    file_path=meta["file_path"],
                    chunk_type=meta["chunk_type"],
                    name=meta.get("name") or None,
                    start_line=int(meta["start_line"]),
                    end_line=int(meta["end_line"]),
                ),
                score=round(1.0 - dist / 2.0, 4),
            ))
        return chunks

    def count(self) -> int:
        return self._collection.count()
