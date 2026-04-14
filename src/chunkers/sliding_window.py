"""
SlidingWindowChunker — groups N consecutive tweets with optional overlap.
"""

import logging

from model.tweet import Tweet
from model.chunk import Chunk
from .base import BaseChunker

logger = logging.getLogger(__name__)


class SlidingWindowChunker(BaseChunker):
    """
    Groups consecutive tweets using a fixed-size sliding window.

    Tweets are sorted chronologically before windowing so the chunk text
    always reads in time order.

    Parameters
    ----------
    window_size : int
        Number of tweets per chunk (default 20).
    stride : int | None
        How many tweets to advance between consecutive windows.
        Defaults to *window_size* (no overlap).
        Set ``stride < window_size`` to produce overlapping chunks, which
        can improve retrieval recall at the cost of more storage and
        embedding calls.

    Examples
    --------
    No overlap (20 tweets / chunk)::

        chunker = SlidingWindowChunker(window_size=20)

    50 % overlap (20 tweet window, 10-tweet stride)::

        chunker = SlidingWindowChunker(window_size=20, stride=10)
    """

    def __init__(self, window_size: int = 20, stride: int | None = None):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        if self.stride < 1:
            raise ValueError("stride must be >= 1")

    def chunk(self, tweets: list[Tweet]) -> list[Chunk]:
        tweets = sorted(tweets, key=lambda t: t.date)
        chunks: list[Chunk] = []

        i = 0
        while i < len(tweets):
            window = tweets[i : i + self.window_size]
            chunks.append(
                Chunk.from_tweets(
                    window,
                    chunk_type="sliding_window",
                    extra_metadata={"window_start_index": i},
                )
            )
            i += self.stride

        logger.info(
            "SlidingWindowChunker: %d tweets → %d chunks "
            "(window_size=%d, stride=%d)",
            len(tweets),
            len(chunks),
            self.window_size,
            self.stride,
        )
        return chunks
