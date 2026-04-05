"""
IdentityChunker — one Chunk per tweet, no grouping.

Useful as a baseline, for debugging, or when you want fine-grained
per-tweet retrieval with full metadata filtering.
"""

import logging

from model.models import Tweet
from .base import BaseChunker, Chunk

logger = logging.getLogger(__name__)


class IdentityChunker(BaseChunker):
    """
    Creates exactly one :class:`Chunk` per tweet.

    No configuration needed.  Each chunk's ``text`` is the tweet's
    ``to_chunk_text()`` output and all standard metadata fields are
    populated as usual, making it fully compatible with vector-store
    filtered queries.
    """

    def chunk(self, tweets: list[Tweet]) -> list[Chunk]:
        chunks = [
            Chunk.from_tweets([tweet], chunk_type="identity")
            for tweet in tweets
            if not tweet.is_empty
        ]
        logger.info("IdentityChunker: %d tweets → %d chunks", len(tweets), len(chunks))
        return chunks
