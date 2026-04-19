"""Helpers for building ChromaDB ``where`` filters from UI selections."""

from __future__ import annotations

from .constants import ALL


def build_where_filter(year: str, platform: str) -> dict | None:
    """Build a ChromaDB metadata filter from the sidebar's Year / Platform picks.

    Returns ``None`` when no filter is active, a single-field dict when exactly
    one is selected, or an ``$and`` compound when both are.
    """
    clauses: list[dict] = []
    if year != ALL:
        clauses.append({"year": {"$eq": int(year)}})
    if platform != ALL:
        clauses.append({"platform": {"$eq": platform}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
