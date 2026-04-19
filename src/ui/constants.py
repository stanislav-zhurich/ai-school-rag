"""UI-level constants shared across tabs."""

from __future__ import annotations

ALL = "All"

YEAR_OPTIONS: list[str] = [ALL] + [str(y) for y in range(2025, 2008, -1)]
PLATFORM_OPTIONS: list[str] = [ALL, "Twitter", "Truth Social"]

DEFAULT_CHUNKS: int = 5
MIN_CHUNKS: int = 1
MAX_CHUNKS: int = 10

# Context-relevance thresholds for the per-query RAG tab.
RELEVANCE_GOOD: float = 0.55
RELEVANCE_FAIR: float = 0.50

# Combined-score thresholds for the per-question eval tab.
EVAL_GOOD: float = 0.70
EVAL_FAIR: float = 0.50


def relevance_quality(avg_score: float) -> str:
    """Label for the average context-relevance score on the RAG tab."""
    if avg_score >= RELEVANCE_GOOD:
        return "🟢 Good"
    if avg_score >= RELEVANCE_FAIR:
        return "🟡 Fair"
    return "🔴 Low"


def eval_quality(combined: float) -> str:
    """Emoji tag used in per-question eval expanders."""
    if combined >= EVAL_GOOD:
        return "🟢"
    if combined >= EVAL_FAIR:
        return "🟡"
    return "🔴"
