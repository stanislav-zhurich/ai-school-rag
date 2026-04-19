"""Streamlit application - DJT Tweet Analyzer.

Three-tab app:

* 📊 Dashboard     - data visualizations over the full tweet corpus
* 💬 RAG Assistant - query the RAG pipeline with context relevance scoring
* 🧪 Evaluation    - run the RAGAS LLM-as-judge evaluation suite

Run from the project root::

    poetry run streamlit run src/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path and relative paths resolve from the project root.
SRC = Path(__file__).parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
os.chdir(ROOT)

import streamlit as st

from ui import (
    get_rag,
    load_tweets,
    render_dashboard_tab,
    render_eval_tab,
    render_rag_tab,
)

st.set_page_config(
    page_title="DJT Tweet Analyzer",
    page_icon="🐦",
    layout="wide",
)

st.title("🐦 Donald Trump's Tweets Analyzer")
st.caption("Explore Donald Trump's posts from 2009 to 2025 and ask questions via RAG.")

tab_dashboard, tab_rag, tab_eval = st.tabs(
    ["📊 Dashboard", "💬 RAG Assistant", "🧪 Evaluation"]
)

with tab_dashboard:
    render_dashboard_tab(load_tweets())

with tab_rag:
    render_rag_tab(get_rag())

with tab_eval:
    render_eval_tab(get_rag())
