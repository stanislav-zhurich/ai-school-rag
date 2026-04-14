"""
Azure OpenAI embeddings via the EPAM AI Proxy (DIAL).

Self-contained: all defaults are declared as class constants.
The only external dependency is the ``OPENAI_API_KEY`` environment variable.
"""

from __future__ import annotations

import os

from openai import AzureOpenAI

from embedder.base_embedder import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
    Embeddings via Azure OpenAI (DIAL proxy).

    Parameters
    ----------
    model : str
        Azure deployment name for the embedding model (default: DEFAULT_MODEL).
    api_key : str | None
        Azure OpenAI API key. Falls back to the ``OPENAI_API_KEY`` env var.
    api_version : str
        Azure OpenAI API version (default: DEFAULT_API_VERSION).
    azure_endpoint : str
        DIAL / Azure endpoint URL (default: DEFAULT_ENDPOINT).
    max_retries : int
        Retries on transient errors with exponential backoff (default 5).
    """

    DEFAULT_MODEL       = "text-embedding-3-small-1"
    DEFAULT_API_VERSION = "2024-10-21"
    DEFAULT_ENDPOINT    = "https://ai-proxy.lab.epam.com"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_version: str = DEFAULT_API_VERSION,
        azure_endpoint: str = DEFAULT_ENDPOINT,
        max_retries: int = 5,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self._client = AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )

    def _raw_embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
