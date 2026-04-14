import chromadb
from typing import Any
from config import CHUNKING_STRATEGY, COLLECTION_NAME
from model.chunk import Chunk
from model.search_result import SearchResult


class ChromaDBStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=f"./chroma_db_{CHUNKING_STRATEGY}")
        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    _CHROMA_BATCH_LIMIT = 5000

    def is_populated(self) -> bool:
        """Return True if the collection already contains at least one document."""
        return self.collection.count() > 0

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their text, metadata and pre-computed embeddings.

        Inserts in batches of at most ``_CHROMA_BATCH_LIMIT`` to stay within
        ChromaDB's hard per-call limit of 5461 records.
        """
        total = len(chunks)
        for start in range(0, total, self._CHROMA_BATCH_LIMIT):
            end = min(start + self._CHROMA_BATCH_LIMIT, total)
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
        """
        Search for the nearest chunks to *query_embedding*.

        Parameters
        ----------
        query_embedding:
            Vector produced by the same embedding model used during ingestion.
        n_results:
            Number of results to return.
        where:
            Optional ChromaDB metadata filter, e.g.
            ``{"year": {"$eq": 2020}}`` or
            ``{"chunk_type": {"$eq": "sliding_window"}}``.
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self.collection.query(**kwargs)

        results: list[SearchResult] = []
        for id_, doc, meta, dist in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            results.append(SearchResult(id=id_, text=doc, metadata=meta, distance=dist))
        return results