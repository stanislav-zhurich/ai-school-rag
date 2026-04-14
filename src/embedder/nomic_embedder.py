"""
Local Nomic embeddings via a running LM Studio (or compatible) server.

Self-contained: connects to DEFAULT_BASE_URL (http://localhost:1234/v1) with
a dummy API key.  No network round-trip to Azure — ideal for offline /
cost-free ingestion runs.
"""

from __future__ import annotations

from openai import OpenAI

from embedder.base_embedder import BaseEmbedder


class NomicEmbedder(BaseEmbedder):
    """
    Embeddings via a locally-running Nomic model served through LM Studio.

    Parameters
    ----------
    model : str
        Model identifier passed to the local server (default: DEFAULT_MODEL).
    base_url : str
        OpenAI-compatible server URL (default: DEFAULT_BASE_URL).
    max_retries : int
        Retries on transient errors with exponential backoff (default 5).
    """

    DEFAULT_MODEL    = "text-embedding-nomic-embed-text-v1.5"
    DEFAULT_BASE_URL = "http://localhost:1234/v1"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = 5,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self._client = OpenAI(
            api_key="lm-model-key",
            base_url=base_url,
        )

    def _raw_embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
