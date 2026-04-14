"""
ChunkerFactory — centralised creation of chunking strategies.

Usage
-----
    chunker = ChunkerFactory.create("sliding_window", window_size=10, stride=5)
    chunker = ChunkerFactory.create("semantic", embed_fn=embedder)
    chunker = ChunkerFactory.create("identity")
"""

from __future__ import annotations

from typing import Any

from chunkers.base import BaseChunker


_STRATEGIES = ("identity", "sliding_window", "semantic")


class ChunkerFactory:
    """Factory that creates :class:`BaseChunker` instances by strategy name."""

    @staticmethod
    def create(strategy: str, **kwargs: Any) -> BaseChunker:
        """
        Instantiate a chunker for the given *strategy*.

        Parameters
        ----------
        strategy : str
            One of ``"identity"``, ``"sliding_window"``, ``"time_window"``,
            ``"semantic"``.
        **kwargs
            Forwarded to the chunker's constructor.

            identity       — no arguments needed
            sliding_window — window_size (int), stride (int | None)
            semantic       — embed_fn (callable), similarity_threshold (float),
                             max_chunk_size (int), min_chunk_size (int)

        Raises
        ------
        ValueError
            If *strategy* is not recognised.
        """
        embed_fn = kwargs.pop("embed_fn", None)

        match strategy:
            case "identity":
                from chunkers.identity import IdentityChunker
                return IdentityChunker(**kwargs)

            case "sliding_window":
                from chunkers.sliding_window import SlidingWindowChunker
                return SlidingWindowChunker(**kwargs)

            case "semantic":
                from chunkers.semantic import SemanticChunker
                if embed_fn is None:
                    raise ValueError(
                        "SemanticChunker requires an 'embed_fn' argument."
                    )
                return SemanticChunker(embed_fn=embed_fn, **kwargs)

            case _:
                raise ValueError(
                    f"Unknown chunking strategy: {strategy!r}. "
                    f"Choose from: {_STRATEGIES}"
                )
