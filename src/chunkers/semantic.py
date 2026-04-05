"""
SemanticChunker — groups tweets by embedding-space similarity.

Algorithm (sequential / greedy):
  1. Sort tweets chronologically.
  2. Embed the first tweet and start a new group.
  3. For each subsequent tweet, embed it and compute the cosine similarity
     against the running centroid of the current group.
  4. If similarity >= threshold AND group size < max_chunk_size → append.
     Otherwise → flush the current group and start a new one.
  5. After all tweets are processed, merge any groups that are smaller than
     min_chunk_size into the previous group.

The embed_fn dependency is injected so the chunker is not tied to any
particular embedding provider (OpenAI, HuggingFace, local, etc.).
"""

import logging
import math
from typing import Protocol, runtime_checkable

from model.models import Tweet
from .base import BaseChunker, Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbedFn(Protocol):
    """Any callable that maps a string to a list of floats."""

    def __call__(self, text: str) -> list[float]: ...


# ---------------------------------------------------------------------------
# Math helpers (no external deps)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _centroid(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class SemanticChunker(BaseChunker):
    """
    Groups tweets by semantic similarity using a caller-supplied embedding
    function.

    Parameters
    ----------
    embed_fn : EmbedFn
        Callable ``(text: str) -> list[float]``.  Typically the
        ``embed`` method of a ``vectorstore.Embedder`` instance.
    similarity_threshold : float
        Cosine-similarity cutoff in [0, 1] (default 0.75).
        Lower  → fewer, broader chunks.
        Higher → more, narrower chunks.
    max_chunk_size : int
        Hard cap on tweets per chunk regardless of similarity (default 30).
    min_chunk_size : int
        Groups smaller than this are merged into the preceding group after
        the main pass (default 3).

    Extra metadata keys
    -------------------
    similarity_threshold (float) – the configured threshold value
    avg_similarity       (float) – mean pairwise similarity within the chunk
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        similarity_threshold: float = 0.75,
        max_chunk_size: int = 30,
        min_chunk_size: int = 3,
    ):
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0, 1]")
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, tweets: list[Tweet]) -> list[Chunk]:
        tweets = sorted(tweets, key=lambda t: t.date)

        # ---- pass 1: greedy sequential grouping -------------------------
        groups: list[list[Tweet]] = []
        group_embeddings: list[list[list[float]]] = []

        current_group: list[Tweet] = []
        current_embeddings: list[list[float]] = []

        for tweet in tweets:
            emb = self.embed_fn(tweet.text)

            if not current_group:
                current_group.append(tweet)
                current_embeddings.append(emb)
                continue

            centroid = _centroid(current_embeddings)
            sim = _cosine_similarity(emb, centroid)
            at_cap = len(current_group) >= self.max_chunk_size

            if sim >= self.similarity_threshold and not at_cap:
                current_group.append(tweet)
                current_embeddings.append(emb)
            else:
                groups.append(current_group)
                group_embeddings.append(current_embeddings)
                current_group = [tweet]
                current_embeddings = [emb]

        if current_group:
            groups.append(current_group)
            group_embeddings.append(current_embeddings)

        # ---- pass 2: merge undersized tail groups -----------------------
        merged_groups: list[list[Tweet]] = []
        merged_embeddings: list[list[list[float]]] = []

        for group, embs in zip(groups, group_embeddings):
            if merged_groups and len(group) < self.min_chunk_size:
                merged_groups[-1].extend(group)
                merged_embeddings[-1].extend(embs)
            else:
                merged_groups.append(group)
                merged_embeddings.append(embs)

        # ---- build Chunk objects ----------------------------------------
        chunks: list[Chunk] = []
        for group, embs in zip(merged_groups, merged_embeddings):
            avg_sim = self._avg_pairwise_similarity(embs)
            chunks.append(
                Chunk.from_tweets(
                    group,
                    chunk_type="semantic",
                    extra_metadata={
                        "similarity_threshold": self.similarity_threshold,
                        "avg_similarity": round(avg_sim, 4),
                    },
                )
            )

        logger.info(
            "SemanticChunker: %d tweets → %d chunks (threshold=%.2f)",
            len(tweets),
            len(chunks),
            self.similarity_threshold,
        )
        return chunks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _avg_pairwise_similarity(embeddings: list[list[float]]) -> float:
        """Mean cosine similarity of all pairs; returns 1.0 for single items."""
        if len(embeddings) < 2:
            return 1.0
        total, count = 0.0, 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                total += _cosine_similarity(embeddings[i], embeddings[j])
                count += 1
        return total / count if count else 1.0
