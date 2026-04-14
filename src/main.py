"""
Entry point for the one-time vector-database ingestion.

Run this script once to download the dataset, chunk the tweets,
compute embeddings, and populate ChromaDB.  The Streamlit app (app.py)
reads from the same database at query time without re-ingesting.

Usage
-----
    poetry run python src/main.py
"""

import config
from chunkers.factory import ChunkerFactory
from embedder import NomicEmbedder, OpenAIEmbedder
from RAG import RAG
from ingestion import IngestionPipeline
from loaders.csv_loader import CSVLoader
from vectorstore import ChromaDBStore


def main() -> None:
    # --- Download raw dataset ------------------------------------------------
    print("Downloading dataset…")
    file_paths = IngestionPipeline.download_kaggle(config.KAGGLE_DATASET_HANDLE)
    print(f"Dataset path: {file_paths[0]}")

    # --- Wire up components --------------------------------------------------
    embedder = OpenAIEmbedder()
    vector_store = ChromaDBStore()

    pipeline = IngestionPipeline(
        loader=CSVLoader(processed_dir=config.PROCESSING_DIR),
        chunker=ChunkerFactory.create(config.CHUNKING_STRATEGY, embed_fn=embedder),
        embedder=embedder,
        vector_store=vector_store,
    )

    # --- Run -----------------------------------------------------------------
    result = pipeline.chunk_embed_store(
        file_paths[0],
        sample=config.MAX_TWEETS,
        random_seed=42,
    )
    print(result)

    rag: RAG = RAG(embedder, vector_store)
    answer, _ = rag.get_answer("What Trump tweets about Putin?")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
