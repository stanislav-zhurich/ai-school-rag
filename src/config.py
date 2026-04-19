"""
Project configuration.

Two dataclass-based groups:

* :class:`AzureSettings`     - Azure OpenAI / DIAL endpoint & credentials
* :class:`PipelineSettings`  - Dataset, chunking, vector-store & UI defaults

Secrets are read lazily via :func:`azure` so importing ``config`` never
raises when ``OPENAI_API_KEY`` is missing; the error is only raised the
first time an Azure-backed component is actually used.

Backwards-compatible module-level constants (``API_KEY``, ``DIAL_URL``,
``CHAT_MODEL``, ``CHUNKING_STRATEGY`` and friends) are still exposed so
existing imports keep working.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


ChunkingStrategy = Literal["identity", "sliding_window", "semantic"]


@dataclass(frozen=True)
class AzureSettings:
    """Azure OpenAI / DIAL endpoint configuration."""

    endpoint: str = "https://ai-proxy.lab.epam.com"
    chat_deployment: str = "gpt-4o"
    api_version: str = "2024-10-21"
    api_key: str = ""


@dataclass(frozen=True)
class PipelineSettings:
    """RAG pipeline, dataset and vector-store configuration."""

    kaggle_dataset_handle: str = "datadrivendecision/trump-tweets-2009-2025"
    chunking_strategy: ChunkingStrategy = "identity"
    max_tweets: int = 10_000
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    collection_name: str = "documents"

    @property
    def chroma_path(self) -> Path:
        """Per-strategy persistent ChromaDB directory (keeps existing layouts)."""
        return Path(f"./chroma_db_{self.chunking_strategy}")


pipeline = PipelineSettings()

_azure: AzureSettings | None = None


def azure() -> AzureSettings:
    """Return Azure settings, loading ``OPENAI_API_KEY`` lazily on first use."""
    global _azure
    if _azure is None:
        try:
            api_key = os.environ["OPENAI_API_KEY"]
        except KeyError as exc:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your environment or .env file."
            ) from exc
        _azure = AzureSettings(api_key=api_key)
    return _azure


# ---------------------------------------------------------------------------
# Backwards-compatible module-level aliases
# ---------------------------------------------------------------------------

KAGGLE_DATASET_HANDLE: str = pipeline.kaggle_dataset_handle
CHUNKING_STRATEGY: ChunkingStrategy = pipeline.chunking_strategy
MAX_TWEETS: int = pipeline.max_tweets
PROCESSING_DIR: str = str(pipeline.processed_dir)
COLLECTION_NAME: str = pipeline.collection_name

DIAL_URL: str = "https://ai-proxy.lab.epam.com"
CHAT_MODEL: str = "gpt-4o"
API_VERSION: str = "2024-10-21"


def __getattr__(name: str) -> str:
    """Lazy attribute access for ``API_KEY`` so imports never hard-fail."""
    if name == "API_KEY":
        return azure().api_key
    raise AttributeError(f"module 'config' has no attribute {name!r}")
