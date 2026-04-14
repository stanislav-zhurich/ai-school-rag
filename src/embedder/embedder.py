"""
OpenAI-backed embeddings for ingestion (batched) and query time (single).

Use the same model for indexed documents and queries unless you have a
specific reason to diverge — OpenAI recommends a single model for both.

``SemanticChunker`` can accept an :class:`OpenAIEmbedder` instance as
``embed_fn`` because :meth:`__call__` delegates to :meth:`embed_query`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import config
from openai import OpenAI, RateLimitError, InternalServerError
from model.chunk import Chunk

_RETRYABLE_ERRORS = (RateLimitError, InternalServerError)


class Embedder:
    """
    Batch and single-text embeddings via the Azure OpenAI API.

    Parameters
    ----------
    model : str
        Embedding model id (default taken from config).
    max_retries : int
        Retries on transient errors (rate-limit, server unavailable)
        with exponential backoff (default 5).
    """

    def __init__(
        self,
        model: str = config.EMBEDDING_MODEL,
        max_retries: int = 5,
    ):
        self.model = model
        self.max_retries = max_retries
        self._client = OpenAI(
                api_key  = 'lm-model-key',
                base_url = f"{config.EMBEDDING_MODEL_URL}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        """Embed a single string (typical RAG user query)."""
        vectors = self._embed_batch([text], batch_size=1)
        return vectors[0]

    def _embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Embed many strings in sub-batches suitable for initial ingestion.

        Preserves input order. Empty *texts* returns ``[]``.
        If a sub-batch fails after all retries it is split in half recursively;
        individual items that still fail are skipped with a warning so one bad
        chunk never aborts the whole pipeline.
        """
        total = len(texts)
        all_embeddings: list[list[float]] = []
        for start in range(0, total, batch_size):
            batch = list(texts[start : start + batch_size])
            print(f"  Embedding items {start + 1}–{min(start + batch_size, total)} / {total}")
            all_embeddings.extend(self._embed_with_fallback(batch))
        return all_embeddings

    def _embed_with_fallback(self, texts: list[str]) -> list[list[float]]:
        """
        Try to embed *texts* as a batch.  On failure, split in half recursively.
        Single items that still fail are skipped (zero vector placeholder logged).
        """
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

    def embed_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        batch_size: int = 128,
    ) -> list[list[float]]:
        """
        Embed ``chunk.text`` for each chunk in order (ingestion helper).
        """
        total = len(chunks)
        print(f"Embedding {total} chunks (batch_size={batch_size})…")
        texts = [c.text for c in chunks]
        embeddings = self._embed_batch(texts, batch_size=batch_size)
        print(f"Embedding complete: {len(embeddings)} / {total} chunks embedded.")
        return embeddings

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_embeddings(self, texts: list[str]) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.embeddings.create(model=self.model, input=texts)
                return [item.embedding for item in response.data]
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                wait = min(2 ** attempt, 30)
                print(f"  {exc.__class__.__name__} — retrying in {wait}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]
