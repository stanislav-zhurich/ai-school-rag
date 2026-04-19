"""Entry point for the one-time vector-database ingestion.

Run this script once to download the dataset, chunk the tweets, compute
embeddings, and populate ChromaDB. The Streamlit app (``app.py``) reads from
the same database at query time without re-ingesting.

Usage
-----
    poetry run python src/main.py
"""

from __future__ import annotations

import config
from chunkers.factory import ChunkerFactory
from embedder import OpenAIEmbedder
from ingestion import IngestionPipeline
from loaders.csv_loader import CSVLoader
from RAG import RAG
from vectorstore import ChromaDBStore


def main() -> None:
    settings = config.pipeline

    print("Downloading dataset…")
    file_paths = IngestionPipeline.download_kaggle(settings.kaggle_dataset_handle)
    print(f"Dataset path: {file_paths[0]}")

    embedder = OpenAIEmbedder()
    vector_store = ChromaDBStore()

    pipeline = IngestionPipeline(
        loader=CSVLoader(processed_dir=str(settings.processed_dir)),
        chunker=ChunkerFactory.create(settings.chunking_strategy, embed_fn=embedder),
        embedder=embedder,
        vector_store=vector_store,
    )

    result = pipeline.chunk_embed_store(
        file_paths[0],
        sample=settings.max_tweets,
        random_seed=42,
    )
    print(result)

    rag = RAG(embedder, vector_store)
    answer, _ = rag.get_answer("What Trump tweets about China?")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
