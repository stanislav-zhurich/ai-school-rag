"""
TimeWindowChunker — groups tweets that fall in the same calendar period.
"""

import logging
from collections import defaultdict

from model.tweet import Tweet
from model.chunk import Chunk
from .base import BaseChunker

logger = logging.getLogger(__name__)

_PERIOD_FORMATS: dict[str, str] = {
    "day":   "%Y-%m-%d",
    "week":  "%G-W%V",   # ISO week (Mon-start)
    "month": "%Y-%m",
    "year":  "%Y",
}


class TimeWindowChunker(BaseChunker):
    """
    Groups tweets that share the same calendar time bucket.

    All tweets in a bucket are sorted chronologically and joined into a
    single :class:`Chunk`.  Empty buckets are never emitted, and buckets
    with fewer tweets than *min_tweets* are silently discarded.

    Parameters
    ----------
    period : str
        Temporal granularity.  One of:
        ``"day"``   → one chunk per calendar day  (``YYYY-MM-DD``)
        ``"week"``  → one chunk per ISO week       (``YYYY-W##``)
        ``"month"`` → one chunk per calendar month (``YYYY-MM``)  ← default
        ``"year"``  → one chunk per calendar year  (``YYYY``)
    min_tweets : int
        Minimum number of tweets required to emit a chunk (default 1).

    Extra metadata keys
    -------------------
    time_bucket (str) – the formatted bucket label, e.g. ``"2020-03"``
    """

    def __init__(self, period: str = "month", min_tweets: int = 1):
        if period not in _PERIOD_FORMATS:
            raise ValueError(
                f"Unknown period {period!r}. "
                f"Choose from: {list(_PERIOD_FORMATS)}"
            )
        self.period = period
        self.min_tweets = min_tweets
        self._fmt = _PERIOD_FORMATS[period]

    def chunk(self, tweets: list[Tweet]) -> list[Chunk]:
        buckets: dict[str, list[Tweet]] = defaultdict(list)
        for tweet in tweets:
            key = tweet.date.strftime(self._fmt)
            buckets[key].append(tweet)

        chunks: list[Chunk] = []
        for key in sorted(buckets):
            group = buckets[key]
            if len(group) < self.min_tweets:
                continue
            chunks.append(
                Chunk.from_tweets(
                    group,
                    chunk_type=f"time_window_{self.period}",
                    extra_metadata={"time_bucket": key},
                )
            )

        logger.info(
            "TimeWindowChunker(%s): %d tweets → %d chunks",
            self.period,
            len(tweets),
            len(chunks),
        )
        return chunks
