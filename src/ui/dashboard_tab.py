"""Dashboard tab: corpus-level visualizations."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model.tweet import Tweet


def _tweets_to_df(tweets: list[Tweet]) -> pd.DataFrame:
    """Flatten ``Tweet`` objects into a dashboard-friendly DataFrame."""
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


def _render_kpis(tweets: list[Tweet], df: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Posts", f"{len(tweets):,}")
    c2.metric("Years Covered", f"{df['year'].min()} – {df['year'].max()}")
    c3.metric("Platforms", int(df["platform"].nunique()))
    c4.metric("Avg Favorites", f"{df['favorite_count'].mean():,.0f}")
    c5.metric("Avg Reposts", f"{df['repost_count'].mean():,.0f}")


def _render_platform_pie(df: pd.DataFrame) -> None:
    st.subheader("Posts by Platform")
    plat = df["platform"].value_counts().reset_index()
    plat.columns = ["platform", "count"]
    fig = px.pie(
        plat,
        names="platform",
        values="count",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=True, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _render_yearly_bars(df: pd.DataFrame) -> None:
    st.subheader("Posts per Year")
    yearly = df.groupby("year").size().reset_index(name="count")
    fig = px.bar(
        yearly,
        x="year",
        y="count",
        color="count",
        color_continuous_scale="Blues",
        labels={"year": "Year", "count": "Posts"},
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _render_engagement_over_time(df: pd.DataFrame) -> None:
    st.subheader("Engagement Over Time")
    eng = (
        df.groupby("year")
        .agg(avg_favorites=("favorite_count", "mean"),
             avg_reposts=("repost_count", "mean"))
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eng["year"], y=eng["avg_favorites"].round(1),
        name="Avg Favorites", mode="lines+markers",
        line=dict(color="#2196F3", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=eng["year"], y=eng["avg_reposts"].round(1),
        name="Avg Reposts", mode="lines+markers",
        line=dict(color="#4CAF50", width=2),
    ))
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Average Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_top_terms(df: pd.DataFrame) -> None:
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Top 20 Hashtags")
        all_tags = [h.lower() for row in df["hashtags"] for h in row]
        if not all_tags:
            st.info("No hashtag data available.")
        else:
            ht_df = pd.DataFrame(
                Counter(all_tags).most_common(20),
                columns=["hashtag", "count"],
            )
            fig = px.bar(
                ht_df.sort_values("count"),
                x="count", y="hashtag",
                orientation="h",
                color="count",
                color_continuous_scale="Reds",
                labels={"hashtag": "", "count": "Count"},
            )
            fig.update_coloraxes(showscale=False)
            fig.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Post Type Breakdown")
        type_counts = {
            "Original": int((~df["is_repost"] & ~df["is_quote"]).sum()),
            "Repost": int(df["is_repost"].sum()),
            "Quote": int(df["is_quote"].sum()),
        }
        type_df = pd.DataFrame(list(type_counts.items()), columns=["type", "count"])
        fig = px.pie(
            type_df, names="type", values="count",
            hole=0.4,
            color_discrete_sequence=["#2196F3", "#FF9800", "#4CAF50"],
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)


def _render_top_mentions(df: pd.DataFrame) -> None:
    st.subheader("Top 20 Mentioned Users")
    all_mentions = [m.lower() for row in df["user_mentions"] for m in row]
    if not all_mentions:
        st.info("No mention data available.")
        return

    men_df = pd.DataFrame(
        Counter(all_mentions).most_common(20),
        columns=["user", "count"],
    )
    fig = px.bar(
        men_df.sort_values("count"),
        x="count", y="user",
        orientation="h",
        color="count",
        color_continuous_scale="Purples",
        labels={"user": "", "count": "Mentions"},
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_dashboard_tab(tweets: list[Tweet]) -> None:
    """Render the entire Dashboard tab."""
    df = _tweets_to_df(tweets)

    _render_kpis(tweets, df)
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        _render_platform_pie(df)
    with col_r:
        _render_yearly_bars(df)

    _render_engagement_over_time(df)
    _render_top_terms(df)
    _render_top_mentions(df)
