"""Evaluation tab: run the full RAGAS evaluation and render the scores."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from eval.eval_dataset import EVAL_SET
from eval.evaluator import EvalResult, evaluate
from RAG import RAG

from .constants import eval_quality

METRIC_COLUMNS: dict[str, str] = {
    "Faithfulness": "faithfulness",
    "Context Precision": "context_precision",
    "Context Recall": "context_recall",
}


def _render_metric_glossary() -> None:
    with st.expander("ℹ️ Metrics explained"):
        st.markdown(
            "**Faithfulness** — are the claims in the answer supported by the retrieved context? "
            "High score = no hallucination.\n\n"
            "**Context Precision** — are the retrieved chunks relevant to the question? "
            "High score = retrieval surfaces useful content.\n\n"
            "**Context Recall** — does the retrieved context contain enough information to answer? "
            "High score = retrieval is sufficiently complete.\n\n"
            "**Combined Score** — simple average of the three metrics above.\n\n"
            "_All metrics are scored 0–1 by the LLM (Azure OpenAI) acting as judge._"
        )


def _results_to_df(results: list[EvalResult]) -> pd.DataFrame:
    """Flatten results plus ``combined_score`` into a DataFrame."""
    rows = []
    for r in results:
        row = asdict(r)
        row["combined_score"] = r.combined_score
        rows.append(row)
    return pd.DataFrame(rows)


def _mean(series: pd.Series) -> float:
    return round(float(series.dropna().mean() or 0.0), 3)


def _render_aggregate(df: pd.DataFrame) -> None:
    st.markdown("### Aggregate Results")
    cols = st.columns(4)
    for col, (label, attr) in zip(cols[:-1], METRIC_COLUMNS.items()):
        col.metric(label, f"{_mean(df[attr]):.3f}")
    cols[-1].metric("Combined Score", f"{_mean(df['combined_score']):.3f}")


def _render_per_category(df: pd.DataFrame) -> None:
    st.markdown("### Results by Category")
    cat = (
        df.groupby("category")
          .agg(
              Questions=("question", "count"),
              Faithfulness=("faithfulness", "mean"),
              **{"Context Precision": ("context_precision", "mean")},
              **{"Context Recall": ("context_recall", "mean")},
              Combined=("combined_score", "mean"),
          )
          .round(3)
          .reset_index()
          .rename(columns={"category": "Category"})
    )
    st.dataframe(cat, use_container_width=True, hide_index=True)


def _render_per_question(results: list[EvalResult]) -> None:
    st.markdown("### Per-question Results")
    for r in results:
        label = eval_quality(r.combined_score)
        header = f"{label} {r.question}  |  Combined: {r.combined_score:.3f}  |  {r.category}"
        with st.expander(header):
            cols = st.columns(4)
            for col, (display, attr) in zip(cols[:-1], METRIC_COLUMNS.items()):
                value = getattr(r, attr)
                col.metric(display, f"{value:.3f}" if value is not None else "—")
            cols[-1].metric("Combined", f"{r.combined_score:.3f}")

            st.markdown("**Generated answer:**")
            st.markdown(r.answer)

            with st.expander("Retrieved context chunks"):
                for i, ctx in enumerate(r.contexts, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.text(ctx)


def render_eval_tab(rag: RAG) -> None:
    """Render the entire Evaluation tab."""
    st.subheader("RAG Pipeline Evaluation")
    st.caption(
        f"Runs {len(EVAL_SET)} predefined questions through the full RAG pipeline "
        "and scores each answer using RAGAS metrics (LLM-as-judge)."
    )

    _render_metric_glossary()

    if not st.button("▶ Run Evaluation", type="primary"):
        return

    with st.spinner(
        f"Running RAGAS evaluation on {len(EVAL_SET)} questions… (this may take a minute)"
    ):
        results: list[EvalResult] = evaluate(rag, EVAL_SET, n_results=5)

    df = _results_to_df(results)

    _render_aggregate(df)
    st.divider()
    _render_per_category(df)
    st.divider()
    _render_per_question(results)
