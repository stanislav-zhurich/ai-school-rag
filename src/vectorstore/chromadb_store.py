"""Persistent ChromaDB vector store used by both ingestion and retrieval."""

from __future__ import annotations

from typing import Any, ClassVar

import chromadb

import config
from model.chunk import Chunk
from model.search_result import SearchResult


class ChromaDBStore:
    """Thin wrapper around a ChromaDB persistent collection.

    The persistent path and collection name are taken from
    :data:`config.pipeline`, and the active chunking strategy is
    recorded in the collection's metadata for debugging.
    """

    CHROMA_BATCH_LIMIT: ClassVar[int] = 5000
    """Chunk-insert batch size (ChromaDB hard limit is 5461 per call)."""

    def __init__(self) -> None:
        settings = config.pipeline
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(
            settings.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "chunking_strategy": settings.chunking_strategy,
            },
        )

    def is_populated(self) -> bool:
        """Return ``True`` if the collection already contains at least one document."""
        return self.collection.count() > 0

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their text, metadata and pre-computed embeddings.

        Inserts in batches of at most :data:`CHROMA_BATCH_LIMIT` to stay within
        ChromaDB's hard per-call limit.
        """
        total = len(chunks)
        for start in range(0, total, self.CHROMA_BATCH_LIMIT):
            end = min(start + self.CHROMA_BATCH_LIMIT, total)
            print(f"Adding chunks {start + 1}–{end} / {total} to ChromaDB…")
            self.collection.add(
                ids=[c.id for c in chunks[start:end]],
                documents=[c.text for c in chunks[start:end]],
                embeddings=embeddings[start:end],
                metadatas=[c.metadata for c in chunks[start:end]],
            )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 3,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the nearest chunks to ``query_embedding``.

        Parameters
        ----------
        query_embedding:
            Vector produced by the same embedding model used during ingestion.
        n_results:
            Number of results to return.
        where:
            Optional ChromaDB metadata filter, e.g. ``{"year": {"$eq": 2020}}``.
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self.collection.query(**kwargs)

        return [
            SearchResult(id=id_, text=doc, metadata=meta, distance=dist)
            for id_, doc, meta, dist in zip(
                raw["ids"][0],
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            )
        ]
