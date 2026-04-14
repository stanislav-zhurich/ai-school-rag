"""
One-time ingestion pipeline: load → chunk → embed → store.

Typical usage
-------------
    pipeline = IngestionPipeline(
        loader=CSVLoader(),
        chunker=ChunkerFactory.create("sliding_window"),
        embedder=NomicEmbedder(),
        vector_store=ChromaDBStore(),
    )
    result = pipeline.chunk_embed_store("data/raw/tweets.csv", sample=2000, random_seed=42)
    print(result)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import kagglehub

from chunkers.base import BaseChunker
from embedder.base_embedder import BaseEmbedder
from loaders.csv_loader import CSVLoader
from model.chunk import Chunk
from model.tweet import Tweet
from vectorstore.chromadb_store import ChromaDBStore


@dataclass
class IngestionResult:
    """Summary statistics returned by :meth:`IngestionPipeline.chunk_embed_store`."""
    tweets_loaded: int
    chunks_created: int
    embeddings_created: int

    @property
    def skipped(self) -> bool:
        """True when ingestion was skipped because the store was already populated."""
        return self.tweets_loaded == 0 and self.chunks_created == 0

    def __str__(self) -> str:
        if self.skipped:
            return "IngestionResult(skipped — vector store already populated)"
        return (
            f"IngestionResult("
            f"tweets={self.tweets_loaded}, "
            f"chunks={self.chunks_created}, "
            f"embeddings={self.embeddings_created})"
        )


class IngestionPipeline:
    """
    Orchestrates the one-time process of filling the vector store.

    Parameters
    ----------
    loader : CSVLoader
        Loads and filters raw tweets from a CSV file.
    chunker : BaseChunker
        Groups tweets into chunks suitable for embedding.
    embedder : BaseEmbedder
        Converts chunk text to embedding vectors.
    vector_store : ChromaDBStore
        Persists chunks and their embeddings for later retrieval.
    """

    def __init__(
        self,
        loader: CSVLoader,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: ChromaDBStore,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_embed_store(self, csv_path: str | Path, **load_kwargs) -> IngestionResult:
        """
        Execute the full load → chunk → embed → store pipeline.

        Parameters
        ----------
        csv_path : str | Path
            Path to the source CSV file.
        **load_kwargs
            Forwarded verbatim to :meth:`CSVLoader.load`
            (e.g. ``sample=2000``, ``random_seed=42``, ``start_date=...``).

        Returns
        -------
        IngestionResult
            Tweet / chunk / embedding counts for the completed run.
        """
        print("=== Ingestion Pipeline ===")

        if self.vector_store.is_populated():
            count = self.vector_store.collection.count()
            print(f"Vector store already contains {count} documents — skipping ingestion.")
            return IngestionResult(tweets_loaded=0, chunks_created=0, embeddings_created=0)

        print("Loading tweets…")
        tweets: list[Tweet] = self.loader.load(csv_path, **load_kwargs)
        print(f"  Tweets loaded    : {len(tweets)}")

        print(f"Chunking tweets…")
        chunks: list[Chunk] = self.chunker.chunk(tweets)
        print(f"  Chunks created   : {len(chunks)}")

        print("Embedding chunks…")
        embeddings: list[list[float]] = self.embedder.embed_chunks(chunks)
        print(f"  Embeddings done  : {len(embeddings)}")

        print("Storing in vector database…")
        self.vector_store.add_chunks(chunks, embeddings)
        print("  Done.")

        return IngestionResult(
            tweets_loaded=len(tweets),
            chunks_created=len(chunks),
            embeddings_created=len(embeddings),
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def download_kaggle(handle: str, output_dir: str = "data/raw") -> list[str]:
        """
        Download a Kaggle dataset and return paths of all regular files.

        Parameters
        ----------
        handle : str
            Kaggle dataset handle, e.g. ``"user/dataset-name"``.
        output_dir : str
            Local directory to store the downloaded files.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = kagglehub.dataset_download(handle=handle, output_dir=output_dir)
        return [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
