"""Streamlit-cached resource factories for the DJT Tweet Analyzer."""

from __future__ import annotations

import os

import streamlit as st

import config
from embedder import OpenAIEmbedder
from ingestion import IngestionPipeline
from loaders.csv_loader import CSVLoader
from model.tweet import Tweet
from RAG import RAG
from vectorstore import ChromaDBStore


@st.cache_data(show_spinner="Loading tweet corpus…")
def load_tweets() -> list[Tweet]:
    """Download (if needed) and parse the Kaggle tweet dataset."""
    settings = config.pipeline
    file_paths = IngestionPipeline.download_kaggle(
        handle=settings.kaggle_dataset_handle,
        output_dir=str(settings.raw_dir),
    )
    if not file_paths:
        raise RuntimeError(
            f"No dataset files found in {settings.raw_dir}. "
            "Check the Kaggle dataset handle and network access."
        )

    loader = CSVLoader(processed_dir=str(settings.processed_dir))
    return loader.load(
        file_paths[0],
        sample=settings.max_tweets,
        random_seed=42,
    )


@st.cache_resource(show_spinner="Initialising RAG pipeline…")
def get_rag() -> RAG:
    """Return a memoised :class:`RAG` wired to the default embedder and store."""
    return RAG(OpenAIEmbedder(), ChromaDBStore())
