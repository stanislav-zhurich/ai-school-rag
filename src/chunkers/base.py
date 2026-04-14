"""
Base abstractions for all chunking strategies.

A Chunk is the unit written to the vector store. It carries:
  - text        : the string that gets embedded
  - tweet_ids   : back-references to source Tweet objects
  - metadata    : flat dict (str/int/float/bool only) so it can be stored
                  directly in ChromaDB and used for filtered queries at
                  retrieval time.

All metadata keys are documented in Chunk.from_tweets so downstream code
can reference them by name without inspecting raw dicts.
"""
from abc import ABC, abstractmethod
from model.tweet import Tweet
from model.chunk import Chunk


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseChunker(ABC):
    """
    Contract for all chunking strategies.

    Implementations receive a list of pre-filtered :class:`Tweet` objects
    and must return a list of :class:`Chunk` objects ready for ingestion
    into the vector store.

    Usage example::

        chunker = SlidingWindowChunker(window_size=20)
        chunks  = chunker.chunk(tweets)
    """

    @abstractmethod
    def chunk(self, tweets: list[Tweet]) -> list[Chunk]:
        """
        Partition *tweets* into chunks.

        Parameters
        ----------
        tweets : list[Tweet]
            Pre-filtered tweet objects (no reposts, min text length applied,
            etc.).  The chunker is free to re-sort them internally.

        Returns
        -------
        list[Chunk]
            Non-empty chunks.  The order is implementation-defined but
            typically chronological.
        """
        ...
