from dataclasses import dataclass, field
from typing import Any
from model.tweet import Tweet
import hashlib
# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A group of tweets packaged for vector-store ingestion.

    Attributes
    ----------
    id : str
        Deterministic identifier derived from the constituent tweet IDs.
    text : str
        The string that will be embedded (all tweets joined with separators).
    tweet_ids : list[str]
        Ordered list of source Tweet IDs for provenance / re-hydration.
    metadata : dict
        Flat mapping of filterable attributes.  All values are
        ``str | int | float | bool`` so they are directly storable in
        ChromaDB.

        Standard keys (always present):
        ================================
        chunk_type      (str)  – strategy name, e.g. "sliding_window"
        tweet_count     (int)  – number of tweets in this chunk
        platform        (str)  – comma-joined unique platform names
        start_date      (str)  – ISO-8601 timestamp of the earliest tweet
        end_date        (str)  – ISO-8601 timestamp of the latest tweet
        year            (int)  – calendar year of the earliest tweet
        month           (str)  – "YYYY-MM" of the earliest tweet
        total_favorites (int)  – sum of favorite counts
        total_reposts   (int)  – sum of repost counts
        hashtags        (str)  – space-joined unique hashtags (may be empty)

    """

    id: str
    text: str
    tweet_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_tweets(
        cls,
        tweets: list[Tweet],
        chunk_type: str,
        id_prefix: str = "chunk",
        extra_metadata: dict[str, Any] | None = None,
    ) -> "Chunk":
        """
        Build a :class:`Chunk` from a list of tweets, computing all
        standard metadata automatically.

        Parameters
        ----------
        tweets : list[Tweet]
            Source tweets (at least one required).
        chunk_type : str
            Label identifying the chunking strategy.
        id_prefix : str
            Prefix for the generated chunk ID.
        extra_metadata : dict | None
            Additional strategy-specific keys merged into metadata.
        """
        if not tweets:
            raise ValueError("Cannot build a Chunk from an empty tweet list.")

        sorted_tweets = sorted(tweets, key=lambda t: t.date)
        ids = [t.id for t in tweets]

        platforms = sorted({t.platform for t in tweets})
        all_hashtags = sorted({h for t in tweets for h in t.hashtags})

        metadata: dict[str, Any] = {
            "chunk_type": chunk_type,
            "tweet_count": len(tweets),
            "platform": ", ".join(platforms),
            "start_date": sorted_tweets[0].date.isoformat(),
            "end_date": sorted_tweets[-1].date.isoformat(),
            "year": sorted_tweets[0].year,
            "month": sorted_tweets[0].month,
            "total_favorites": sum(t.favorite_count for t in tweets),
            "total_reposts": sum(t.repost_count for t in tweets),
            "hashtags": " ".join(all_hashtags),
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        text = "\n\n---\n\n".join(t.to_chunk_text() for t in sorted_tweets)

        return cls(
            id=_make_chunk_id(ids, prefix=id_prefix),
            text=text,
            tweet_ids=ids,
            metadata=metadata,
        )
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk_id(tweet_ids: list[str], prefix: str = "chunk") -> str:
    """Deterministic, collision-resistant chunk ID from constituent tweet IDs."""
    payload = ",".join(sorted(tweet_ids))
    digest = hashlib.sha1(payload.encode()).hexdigest()[:12]
    return f"{prefix}_{digest}"