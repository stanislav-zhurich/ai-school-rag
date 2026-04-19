"""RAG Assistant tab: ask a question, see an answer, inspect retrieved chunks."""

from __future__ import annotations

import streamlit as st

from model.search_result import SearchResult
from RAG import RAG

from .constants import (
    DEFAULT_CHUNKS,
    MAX_CHUNKS,
    MIN_CHUNKS,
    PLATFORM_OPTIONS,
    RELEVANCE_FAIR,
    YEAR_OPTIONS,
    relevance_quality,
)
from .filters import build_where_filter


def _render_filters() -> tuple[str, str, int]:
    """Render inline filters and return the user's selections."""
    f1, f2, f3 = st.columns([2, 2, 3])
    with f1:
        year = st.selectbox("Year", YEAR_OPTIONS)
    with f2:
        platform = st.selectbox("Platform", PLATFORM_OPTIONS)
    with f3:
        n_results = st.slider(
            "Chunks to retrieve",
            min_value=MIN_CHUNKS, max_value=MAX_CHUNKS, value=DEFAULT_CHUNKS,
        )
    return year, platform, n_results


def _render_relevance(results: list[SearchResult]) -> list[float]:
    """Show per-chunk and average context-relevance metrics. Returns the scores."""
    scores = [max(0.0, 1.0 - r.distance) for r in results]
    avg = sum(scores) / len(scores)

    st.markdown("### Evaluation: Context Relevance Score")
    st.caption(
        "Cosine similarity between the query embedding and each retrieved chunk "
        "(1.0 = identical, 0.0 = unrelated). Higher average → more relevant context."
    )

    cols = st.columns(len(results) + 1)
    for i, (r, score) in enumerate(zip(results, scores)):
        cols[i].metric(
            label=f"Chunk {i + 1}",
            value=f"{score:.3f}",
            help=r.metadata.get("start_date", "")[:10],
        )
    cols[-1].metric("Average", f"{avg:.3f}", delta=relevance_quality(avg))

    if avg < RELEVANCE_FAIR:
        st.warning(
            "⚠️ The retrieved context has low relevance to this query. "
            "The answer may be unreliable or the topic may not be covered in the dataset.",
            icon=None,
        )
    return scores


def _render_chunks(results: list[SearchResult], scores: list[float]) -> None:
    st.markdown("### Retrieved Context Chunks")
    for i, (r, score) in enumerate(zip(results, scores)):
        date = r.metadata.get("start_date", "")[:10]
        platform = r.metadata.get("platform", "")
        with st.expander(
            f"Chunk {i + 1}  |  Relevance: {score:.3f}  |  {date}  |  {platform}"
        ):
            st.text(r.text)
            with st.expander("Metadata"):
                st.json(r.metadata)


def render_rag_tab(rag: RAG) -> None:
    """Render the entire RAG Assistant tab."""
    st.subheader("Ask questions about Trump's tweets")

    year, platform, n_results = _render_filters()
    where = build_where_filter(year, platform)

    with st.form("rag_form"):
        query = st.text_input(
            "Your question",
            placeholder="e.g. What did Trump tweet about NATO?",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if not (submitted and query.strip()):
        return

    with st.spinner("Retrieving context and generating answer…"):
        answer, results = rag.get_answer(query, n_results=n_results, where=where)

    st.markdown("### Answer")
    st.markdown(answer)
    st.divider()

    if not results:
        st.warning(
            "No chunks were retrieved. Make sure you have indexed data by running "
            "`poetry run python src/main.py` first."
        )
        return

    scores = _render_relevance(results)
    _render_chunks(results, scores)
