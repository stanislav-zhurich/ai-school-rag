"""
Streamlit application — DJT Tweet Analyzer
==========================================
Two-tab app:
  📊 Dashboard  – data visualizations over the full tweet corpus
  💬 RAG Assistant – query the RAG pipeline with context relevance scoring

Run from the project root:
    poetry run streamlit run src/app.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

# Ensure src/ is on the path so all project imports resolve,
# and set CWD to project root so relative paths (data/, chroma_db/) work.
SRC = Path(__file__).parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
os.chdir(ROOT)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from embedder import OpenAIEmbedder, OpenAIEmbedder
from loaders.csv_loader import CSVLoader
from model.tweet import Tweet
from RAG import RAG
from vectorstore.chromadb_store import ChromaDBStore
from eval.eval_dataset import EVAL_SET
from eval.evaluator import EvalResult, evaluate
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DJT Tweet Analyzer",
    page_icon="🐦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading tweet corpus…")
def load_tweets() -> list[Tweet]:
    import kagglehub
    os.makedirs("data/raw", exist_ok=True)
    path = kagglehub.dataset_download(
        config.KAGGLE_DATASET_HANDLE, output_dir="data/raw"
    )
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    ]
    loader = CSVLoader(processed_dir=config.PROCESSING_DIR)
    return loader.load(files[0], sample=config.MAX_TWEETS, random_seed=42)


@st.cache_resource(show_spinner="Initialising RAG pipeline…")
def get_rag() -> RAG:
    return RAG(OpenAIEmbedder(), ChromaDBStore())


def tweets_to_df(tweets: list[Tweet]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "year": t.year,
            "month": t.month,
            "platform": t.platform,
            "favorite_count": t.favorite_count,
            "repost_count": t.repost_count,
            "word_count": t.word_count,
            "hashtags": t.hashtags,
            "user_mentions": t.user_mentions,
            "is_repost": t.repost_flag,
            "is_quote": t.quote_flag,
            "is_deleted": t.deleted_flag,
            "has_media": t.media_count > 0,
        }
        for t in tweets
    ])


def build_where_filter(year: str, platform: str) -> dict | None:
    """Construct a ChromaDB $and / single filter from sidebar selections."""
    filters: dict = {}
    if year != "All":
        filters["year"] = {"$eq": int(year)}
    if platform != "All":
        filters["platform"] = {"$eq": platform}

    if not filters:
        return None
    if len(filters) == 1:
        return filters
    return {"$and": [{k: v} for k, v in filters.items()]}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.title("🐦 Donald Trump's Tweets Analyzer")
st.caption("Explore Donald Trump's posts from 2009 to 2025 and ask questions via RAG.")

tab_dashboard, tab_rag, tab_eval = st.tabs(["📊 Dashboard", "💬 RAG Assistant", "🧪 Evaluation"])

# ===========================================================================
# TAB 1 — DASHBOARD
# ===========================================================================

with tab_dashboard:
    tweets = load_tweets()
    df = tweets_to_df(tweets)

    # KPI metrics ---------------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Posts", f"{len(tweets):,}")
    c2.metric("Years Covered", f"{df['year'].min()} – {df['year'].max()}")
    c3.metric("Platforms", int(df["platform"].nunique()))
    c4.metric("Avg Favorites", f"{df['favorite_count'].mean():,.0f}")
    c5.metric("Avg Reposts", f"{df['repost_count'].mean():,.0f}")

    st.divider()

    # Row 1: Platform distribution + Posts per year ----------------------------
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Posts by Platform")
        plat = df["platform"].value_counts().reset_index()
        plat.columns = ["platform", "count"]
        fig_pie = px.pie(
            plat,
            names="platform",
            values="count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=True, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.subheader("Posts per Year")
        yearly = df.groupby("year").size().reset_index(name="count")
        fig_yearly = px.bar(
            yearly,
            x="year",
            y="count",
            color="count",
            color_continuous_scale="Blues",
            labels={"year": "Year", "count": "Posts"},
        )
        fig_yearly.update_coloraxes(showscale=False)
        fig_yearly.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_yearly, use_container_width=True)

    # Row 2: Engagement over time (line) ---------------------------------------
    st.subheader("Engagement Over Time")
    eng = (
        df.groupby("year")
        .agg(avg_favorites=("favorite_count", "mean"), avg_reposts=("repost_count", "mean"))
        .reset_index()
    )
    fig_eng = go.Figure()
    fig_eng.add_trace(go.Scatter(
        x=eng["year"], y=eng["avg_favorites"].round(1),
        name="Avg Favorites", mode="lines+markers",
        line=dict(color="#2196F3", width=2),
    ))
    fig_eng.add_trace(go.Scatter(
        x=eng["year"], y=eng["avg_reposts"].round(1),
        name="Avg Reposts", mode="lines+markers",
        line=dict(color="#4CAF50", width=2),
    ))
    fig_eng.update_layout(
        xaxis_title="Year",
        yaxis_title="Average Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig_eng, use_container_width=True)

    # Row 3: Top hashtags + Post type breakdown --------------------------------
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.subheader("Top 20 Hashtags")
        all_tags = [h.lower() for row in df["hashtags"] for h in row]
        if all_tags:
            top_tags = Counter(all_tags).most_common(20)
            ht_df = pd.DataFrame(top_tags, columns=["hashtag", "count"])
            fig_ht = px.bar(
                ht_df.sort_values("count"),
                x="count",
                y="hashtag",
                orientation="h",
                color="count",
                color_continuous_scale="Reds",
                labels={"hashtag": "", "count": "Count"},
            )
            fig_ht.update_coloraxes(showscale=False)
            fig_ht.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig_ht, use_container_width=True)
        else:
            st.info("No hashtag data available.")

    with col_r2:
        st.subheader("Post Type Breakdown")
        type_counts = {
            "Original": int((~df["is_repost"] & ~df["is_quote"]).sum()),
            "Repost": int(df["is_repost"].sum()),
            "Quote": int(df["is_quote"].sum()),
        }
        type_df = pd.DataFrame(list(type_counts.items()), columns=["type", "count"])
        fig_type = px.pie(
            type_df,
            names="type",
            values="count",
            hole=0.4,
            color_discrete_sequence=["#2196F3", "#FF9800", "#4CAF50"],
        )
        fig_type.update_traces(textposition="inside", textinfo="percent+label")
        fig_type.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_type, use_container_width=True)

    # Row 4: Top mentioned users -----------------------------------------------
    st.subheader("Top 20 Mentioned Users")
    all_mentions = [m.lower() for row in df["user_mentions"] for m in row]
    if all_mentions:
        top_mentions = Counter(all_mentions).most_common(20)
        men_df = pd.DataFrame(top_mentions, columns=["user", "count"])
        fig_men = px.bar(
            men_df.sort_values("count"),
            x="count",
            y="user",
            orientation="h",
            color="count",
            color_continuous_scale="Purples",
            labels={"user": "", "count": "Mentions"},
        )
        fig_men.update_coloraxes(showscale=False)
        fig_men.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_men, use_container_width=True)
    else:
        st.info("No mention data available.")

# ===========================================================================
# TAB 2 — RAG ASSISTANT
# ===========================================================================

with tab_rag:
    st.subheader("Ask questions about Trump's tweets")

    # Inline filters -----------------------------------------------------------
    f_col1, f_col2, f_col3 = st.columns([2, 2, 3])
    with f_col1:
        year_opts = ["All"] + [str(y) for y in range(2025, 2008, -1)]
        selected_year = st.selectbox("Year", year_opts)
    with f_col2:
        platform_opts = ["All", "Twitter", "Truth Social"]
        selected_platform = st.selectbox("Platform", platform_opts)
    with f_col3:
        n_results = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)

    where = build_where_filter(selected_year, selected_platform)

    with st.form("rag_form"):
        query = st.text_input(
            "Your question",
            placeholder="e.g. What did Trump tweet about NATO?",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted and query.strip():
        rag = get_rag()

        with st.spinner("Retrieving context and generating answer…"):
            answer, results = rag.get_answer(query, n_results=n_results, where=where)

        # Answer ---------------------------------------------------------------
        st.markdown("### Answer")
        st.markdown(answer)

        st.divider()

        # Evaluation metric: Context Relevance Score ---------------------------
        if results:
            relevance_scores = [max(0.0, 1.0 - r.distance) for r in results]
            avg_score = sum(relevance_scores) / len(relevance_scores)

            st.markdown("### Evaluation: Context Relevance Score")
            st.caption(
                "Cosine similarity between the query embedding and each retrieved chunk "
                "(1.0 = identical, 0.0 = unrelated). Higher average → more relevant context."
            )

            score_cols = st.columns(len(results) + 1)
            for i, (r, score) in enumerate(zip(results, relevance_scores)):
                score_cols[i].metric(
                    label=f"Chunk {i + 1}",
                    value=f"{score:.3f}",
                    help=r.metadata.get("start_date", "")[:10],
                )

            quality = "🟢 Good" if avg_score >= 0.55 else "🟡 Fair" if avg_score >= 0.5 else "🔴 Low"
            score_cols[-1].metric("Average", f"{avg_score:.3f}", delta=quality)

            if avg_score < 0.5:
                st.warning(
                    "⚠️ The retrieved context has low relevance to this query. "
                    "The answer may be unreliable or the topic may not be covered in the dataset.",
                    icon=None,
                )

            # Retrieved chunks (expandable) ------------------------------------
            st.markdown("### Retrieved Context Chunks")
            for i, (r, score) in enumerate(zip(results, relevance_scores)):
                date = r.metadata.get("start_date", "")[:10]
                platform = r.metadata.get("platform", "")
                with st.expander(
                    f"Chunk {i + 1}  |  Relevance: {score:.3f}  |  {date}  |  {platform}"
                ):
                    st.text(r.text)
                    with st.expander("Metadata"):
                        st.json(r.metadata)
        else:
            st.warning(
                "No chunks were retrieved. Make sure you have indexed data by running "
                "`poetry run python src/main.py` first."
            )

# ===========================================================================
# TAB 3 — EVALUATION
# ===========================================================================

with tab_eval:
    st.subheader("RAG Pipeline Evaluation")
    st.caption(
        f"Runs {len(EVAL_SET)} predefined questions through the full RAG pipeline "
        "and scores each answer using RAGAS metrics (LLM-as-judge)."
    )

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

    if st.button("▶ Run Evaluation", type="primary"):
        rag = get_rag()

        with st.spinner(f"Running RAGAS evaluation on {len(EVAL_SET)} questions… (this may take a minute)"):
            eval_results: list[EvalResult] = evaluate(rag, EVAL_SET, n_results=5)

        # --- Aggregate scores -------------------------------------------------
        def _avg(attr: str) -> float:
            vals = [getattr(r, attr) for r in eval_results if getattr(r, attr) is not None]
            return round(sum(vals) / len(vals), 3) if vals else 0.0

        st.markdown("### Aggregate Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Faithfulness",      f"{_avg('faithfulness'):.3f}",      help="No hallucination score")
        m2.metric("Context Precision", f"{_avg('context_precision'):.3f}", help="Retrieval relevance score")
        m3.metric("Context Recall",    f"{_avg('context_recall'):.3f}",    help="Retrieval completeness score")
        m4.metric("Combined Score",    f"{_avg('combined_score'):.3f}")

        st.divider()

        # --- Per-category summary ---------------------------------------------
        st.markdown("### Results by Category")
        categories = sorted({r.category for r in eval_results})
        cat_rows = []
        for cat in categories:
            cat_res = [r for r in eval_results if r.category == cat]
            cat_rows.append({
                "Category":          cat,
                "Questions":         len(cat_res),
                "Faithfulness":      round(sum(r.faithfulness      or 0 for r in cat_res) / len(cat_res), 3),
                "Context Precision": round(sum(r.context_precision or 0 for r in cat_res) / len(cat_res), 3),
                "Context Recall":    round(sum(r.context_recall    or 0 for r in cat_res) / len(cat_res), 3),
                "Combined":          round(sum(r.combined_score        for r in cat_res) / len(cat_res), 3),
            })
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

        st.divider()

        # --- Per-question detail ----------------------------------------------
        st.markdown("### Per-question Results")
        for r in eval_results:
            score_label = "🟢" if r.combined_score >= 0.7 else "🟡" if r.combined_score >= 0.5 else "🔴"
            with st.expander(
                f"{score_label} {r.question}  |  Combined: {r.combined_score:.3f}  |  {r.category}"
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Faithfulness",      f"{r.faithfulness:.3f}"      if r.faithfulness      is not None else "—")
                c2.metric("Context Precision", f"{r.context_precision:.3f}" if r.context_precision is not None else "—")
                c3.metric("Context Recall",    f"{r.context_recall:.3f}"    if r.context_recall    is not None else "—")
                c4.metric("Combined",          f"{r.combined_score:.3f}")

                st.markdown("**Generated answer:**")
                st.markdown(r.answer)

                with st.expander("Retrieved context chunks"):
                    for i, ctx in enumerate(r.contexts, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.text(ctx)
