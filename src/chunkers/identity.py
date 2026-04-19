"""
IdentityChunker — one Chunk per tweet, no grouping.

Useful as a baseline, for debugging, or when you want fine-grained
per-tweet retrieval with full metadata filtering.
"""

from model.tweet import Tweet
from model.chunk import Chunk
from .base import BaseChunker


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
        print(f"IdentityChunker: {len(tweets)} tweets → {len(chunks)} chunks")
        return chunks
