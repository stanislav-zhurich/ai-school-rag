"""
LLMChunker — uses an LLM to group tweets by theme / topic.

The LLM receives batches of up to *batch_size* tweets and is asked to
return a JSON grouping.  Each group becomes one Chunk with an LLM-assigned
``theme`` label stored in metadata.

Any tweet IDs the LLM omits or returns invalid JSON for are safely
recovered: they fall back to individual single-tweet chunks so no data is
lost.

Prompt design
-------------
The system prompt is minimal and deterministic (temperature=0) to keep
groupings stable across reruns with the same data.  ``response_format`` is
set to ``json_object`` (supported by gpt-4o / gpt-4o-mini) to guarantee
parseable output.
"""

import json
import logging
from typing import Any

from model.tweet import Tweet
from model.chunk import Chunk
from .base import BaseChunker

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a text-analysis assistant. You will receive a numbered list of social
media posts with their IDs. Your task is to group them by shared topic or theme.

Return ONLY a JSON object with a single key "groups" whose value is an array.
Each element of the array must be an object with exactly two keys:
  "theme" : a short label for the group  (≤ 6 words)
  "ids"   : an array of post ID strings belonging to this group

Rules:
- Every input ID must appear in exactly one group — no omissions, no duplicates.
- Aim for groups of 3–15 posts.
- If a post does not fit any theme, place it in its own single-item group.
- Do NOT include any text outside the JSON object.\
"""


class LLMChunker(BaseChunker):
    """
    Groups tweets by theme using an LLM chat-completion call.

    Parameters
    ----------
    client : openai.OpenAI
        An initialised OpenAI (or compatible) client.
    model : str
        Chat-completion model name (default ``"gpt-4o-mini"``).
    batch_size : int
        Number of tweets sent to the LLM in one request (default 50).
        Keep ≤ 100 to stay within practical context limits.

    Extra metadata keys
    -------------------
    theme (str) – LLM-assigned theme label for the group
    """

    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o-mini",
        batch_size: int = 50,
    ):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def chunk(self, tweets: list[Tweet]) -> list[Chunk]:
        tweets = sorted(tweets, key=lambda t: t.date)
        all_chunks: list[Chunk] = []

        for batch_start in range(0, len(tweets), self.batch_size):
            batch = tweets[batch_start : batch_start + self.batch_size]
            groups = self._group_batch(batch)
            id_to_tweet = {t.id: t for t in batch}

            for theme, ids in groups.items():
                group_tweets = [id_to_tweet[i] for i in ids if i in id_to_tweet]
                if not group_tweets:
                    continue
                all_chunks.append(
                    Chunk.from_tweets(
                        group_tweets,
                        chunk_type="llm",
                        extra_metadata={"theme": theme},
                    )
                )

        logger.info(
            "LLMChunker: %d tweets → %d chunks (model=%s, batch_size=%d)",
            len(tweets),
            len(all_chunks),
            self.model,
            self.batch_size,
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _group_batch(self, tweets: list[Tweet]) -> dict[str, list[str]]:
        """
        Call the LLM for one batch and return ``{theme: [tweet_id, ...]}``.
        Falls back to one-tweet-per-chunk if the call or parse fails.
        """
        user_content = "\n".join(
            f"ID:{t.id} | {t.date.strftime('%Y-%m-%d')} | {t.text[:300]}"
            for t in tweets
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw = response.choices[0].message.content or "{}"
            return self._parse_response(raw, tweets)
        except Exception as exc:
            logger.warning(
                "LLMChunker: LLM call failed (%s) — falling back to "
                "one-tweet chunks for this batch.",
                exc,
            )
            return {t.id: [t.id] for t in tweets}

    def _parse_response(
        self,
        raw: str,
        tweets: list[Tweet],
    ) -> dict[str, list[str]]:
        """
        Parse the LLM JSON response into ``{theme: [tweet_id, ...]}``.

        Handles:
        - ``{"groups": [...]}``  (expected)
        - ``[...]``              (bare array fallback)
        - Any missing / duplicated IDs are reconciled so every tweet is
          assigned to exactly one group.
        """
        all_ids = {t.id for t in tweets}
        result: dict[str, list[str]] = {}
        seen_ids: set[str] = set()

        try:
            data = json.loads(raw)
            groups_list = (
                data.get("groups", [])
                if isinstance(data, dict)
                else data
            )
            for item in groups_list:
                theme = str(item.get("theme", "misc")).strip() or "misc"
                raw_ids = [str(i) for i in item.get("ids", [])]
                valid_ids = [
                    i for i in raw_ids
                    if i in all_ids and i not in seen_ids
                ]
                if valid_ids:
                    result[theme] = valid_ids
                    seen_ids.update(valid_ids)

        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            logger.warning("LLMChunker: failed to parse LLM response: %s", exc)

        # Recover any IDs the LLM missed
        for tweet in tweets:
            if tweet.id not in seen_ids:
                result[f"ungrouped_{tweet.id}"] = [tweet.id]

        return result
