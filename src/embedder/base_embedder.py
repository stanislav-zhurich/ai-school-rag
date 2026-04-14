"""
Abstract base class for all embedder implementations.

Concrete subclasses only need to implement :meth:`_raw_embed`, which makes
the actual API / model call for a flat list of strings.  All batching, retry,
and fallback logic lives here so it is shared across every backend.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence

from openai import RateLimitError, InternalServerError

from model.chunk import Chunk

_RETRYABLE_ERRORS = (RateLimitError, InternalServerError)


class BaseEmbedder(ABC):
    """
    Shared embedding interface with batching and retry logic.

    Subclasses must set ``self.model`` and ``self.max_retries`` in
    ``__init__`` and implement :meth:`_raw_embed`.
    """

    model: str
    max_retries: int

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, text: str) -> list[float]:
        """Make the embedder callable — required by SemanticChunker."""
        return self.embed_query(text)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single string (typical RAG query)."""
        vectors = self._embed_batch([text], batch_size=1)
        if not vectors:
            raise RuntimeError(
                f"Embedding failed for query text — the {self.__class__.__name__} "
                "returned no result. Check that the embedding backend is reachable."
            )
        return vectors[0]

    def embed_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        batch_size: int = 128,
    ) -> list[list[float]]:
        """Embed ``chunk.text`` for every chunk (ingestion helper)."""
        total = len(chunks)
        print(f"Embedding {total} chunks (batch_size={batch_size})…")
        texts = [c.text for c in chunks]
        embeddings = self._embed_batch(texts, batch_size=batch_size)
        print(f"Embedding complete: {len(embeddings)} / {total} chunks embedded.")
        return embeddings

    # ------------------------------------------------------------------
    # Batching / fallback (shared)
    # ------------------------------------------------------------------

    def _embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Embed *texts* in sub-batches.

        Preserves input order.  Empty *texts* returns ``[]``.
        Failed sub-batches are split in half recursively; items that
        still fail individually are skipped with a warning.
        """
        total = len(texts)
        all_embeddings: list[list[float]] = []
        for start in range(0, total, batch_size):
            batch = list(texts[start : start + batch_size])
            print(f"  Embedding items {start + 1}–{min(start + batch_size, total)} / {total}")
            all_embeddings.extend(self._embed_with_fallback(batch))
        return all_embeddings

    def _embed_with_fallback(self, texts: list[str]) -> list[list[float]]:
        """Try to embed *texts* as a batch; split in half on failure."""
        try:
            return self._create_embeddings(texts)
        except Exception as exc:
            if len(texts) == 1:
                print(f"  WARNING: skipping 1 item after repeated failures ({exc.__class__.__name__}: {exc})")
                return []
            mid = len(texts) // 2
            print(f"  Batch of {len(texts)} failed — splitting into two halves and retrying…")
            left  = self._embed_with_fallback(texts[:mid])
            right = self._embed_with_fallback(texts[mid:])
            return left + right

    def _create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Retry wrapper around :meth:`_raw_embed` with exponential backoff."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._raw_embed(texts)
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                wait = min(2 ** attempt, 30)
                print(f"  {exc.__class__.__name__} — retrying in {wait}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Abstract — implemented by each backend
    # ------------------------------------------------------------------

    @abstractmethod
    def _raw_embed(self, texts: list[str]) -> list[list[float]]:
        """
        Make a single embedding call for *texts* (no retry, no batching).

        Returns a list of float vectors in the same order as *texts*.
        Raise on any error — the base class handles retries and fallback.
        """
