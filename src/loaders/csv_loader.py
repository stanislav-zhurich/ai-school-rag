"""
CSVLoader  –  loads the DJT tweets CSV into Tweet objects.

Parsing strategy
----------------
* Reads the raw CSV with the stdlib `csv` module (no extra deps).
* List-valued columns (hashtags, urls, user_mentions, media_urls) are stored
  as whitespace-separated tokens in the CSV; they are split and stripped here.
* Boolean columns ("True"/"False" strings) are converted to Python booleans.
* The `date` column carries a UTC-offset ISO-8601 timestamp which is parsed
  with `datetime.fromisoformat`.

Filtering
---------
All filters are applied during the load pass to avoid building a large
intermediate list.  Supported filters:
  * start_date / end_date  – inclusive datetime range
  * platforms              – whitelist of platform strings (case-insensitive)
  * exclude_reposts        – drop rows where repost_flag is True
  * exclude_deleted        – drop rows where deleted_flag is True
  * min_text_length        – drop rows whose text is shorter than N characters
  * max_rows               – stop after loading N tweets (after filtering)

Caching
-------
If *processed_dir* is provided, the loader writes the parsed result as a JSON
file on the first run and reads from it on subsequent calls, bypassing CSV
parsing entirely.  The cache key is based on the CSV filename.
"""

import csv
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from model.tweet import Tweet

_BOOL_MAP = {"true": True, "false": False, "1": True, "0": False}


def _parse_bool(value: str) -> bool:
    return _BOOL_MAP.get(value.strip().lower(), False)


def _parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _parse_list(value: str) -> list[str]:
    """Split a whitespace-separated multi-value cell, ignoring empty tokens."""
    return [tok for tok in value.split() if tok]


def _parse_date(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; fall back to UTC-aware epoch on failure."""
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        print(f"Could not parse date {value}, using epoch")
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


class CSVLoader:
    """
    Loads DJT tweet records from a CSV file into :class:`Tweet` objects.

    Usage
    -----
    loader = CSVLoader(processed_dir="data/processed")
    tweets = loader.load(
        "data/raw/djt_posts_dec2025.csv",
        start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        exclude_reposts=True,
    )
    """

    def __init__(
        self,
        processed_dir: str | Path | None = None,
        min_text_length: int = 10,
    ):
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.min_text_length = min_text_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        csv_path: str | Path,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        platforms: list[str] | None = None,
        exclude_reposts: bool = False,
        exclude_deleted: bool = False,
        max_rows: int | None = None,
        sample: int | None = None,
        random_seed: int | None = None,
    ) -> list[Tweet]:
        """
        Parse *csv_path* and return a filtered list of :class:`Tweet` objects.

        Parameters
        ----------
        csv_path:        Path to the raw CSV file.
        start_date:      Keep only tweets on or after this datetime (tz-aware).
        end_date:        Keep only tweets on or before this datetime (tz-aware).
        platforms:       Whitelist of platform names; ``None`` keeps all.
        exclude_reposts: When *True*, drop reposted entries.
        exclude_deleted: When *True*, drop deleted entries.
        max_rows:        Cap the number of tweets returned sequentially (first N
                         tweets that pass filters).
        sample:          Return a random subset of *sample* tweets from the full
                         filtered pool.  Unlike *max_rows*, this draws from the
                         entire dataset rather than stopping at the first N.
                         Mutually exclusive with *max_rows*.
        random_seed:     Seed for the random sampler so results are reproducible.
                         Only used when *sample* is set.
        """
        if max_rows and sample:
            raise ValueError("max_rows and sample are mutually exclusive — use one or the other.")

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        cache_path = self._cache_path(csv_path)
        if cache_path and cache_path.exists():
            print(f"Loading tweets from cache: {cache_path}")
            tweets = self._load_cache(cache_path)
        else:
            print(f"Parsing CSV: {csv_path}")
            tweets = self._parse_csv(csv_path)
            if cache_path:
                self._save_cache(tweets, cache_path)

        platforms_lower = {p.lower() for p in platforms} if platforms else None

        filtered: list[Tweet] = []
        for t in tweets:
            if exclude_reposts and t.repost_flag:
                continue
            if exclude_deleted and t.deleted_flag:
                continue
            if len(t.text.strip()) < self.min_text_length:
                continue
            if start_date and t.date < start_date:
                continue
            if end_date and t.date > end_date:
                continue
            if platforms_lower and t.platform.lower() not in platforms_lower:
                continue
            filtered.append(t)
            if max_rows and len(filtered) >= max_rows:
                break

        if sample is not None:
            rng = random.Random(random_seed)
            filtered = rng.sample(filtered, min(sample, len(filtered)))

        print(
            "Loaded %d tweets from %s (%d after filtering%s)",
            len(tweets),
            csv_path.name,
            len(filtered),
            f", {len(filtered)} sampled" if sample is not None else "",
        )
        return filtered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self, csv_path: Path) -> Path | None:
        if self.processed_dir is None:
            return None
        return self.processed_dir / (csv_path.stem + "_tweets.json")

    def _parse_csv(self, csv_path: Path) -> list[Tweet]:
        tweets: list[Tweet] = []
        skipped = 0
        with csv_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                tweet = self._parse_row(row)
                if tweet is not None:
                    tweets.append(tweet)
                else:
                    skipped += 1
        if skipped:
            print(f"Skipped {skipped} malformed rows")
        return tweets

    def _parse_row(self, row: dict) -> Tweet | None:
        try:
            return Tweet(
                id=row["id"].strip(),
                date=_parse_date(row["date"]),
                platform=row["platform"].strip(),
                handle=row["handle"].strip(),
                text=row["text"].strip(),
                favorite_count=_parse_int(row.get("favorite_count", "0")),
                repost_count=_parse_int(row.get("repost_count", "0")),
                quote_flag=_parse_bool(row.get("quote_flag", "false")),
                repost_flag=_parse_bool(row.get("repost_flag", "false")),
                deleted_flag=_parse_bool(row.get("deleted_flag", "false")),
                word_count=_parse_int(row.get("word_count", "0")),
                hashtags=_parse_list(row.get("hashtags", "")),
                urls=_parse_list(row.get("urls", "")),
                user_mentions=_parse_list(row.get("user_mentions", "")),
                media_count=_parse_int(row.get("media_count", "0")),
                media_urls=_parse_list(row.get("media_urls", "")),
                post_url=row.get("post_url", "").strip(),
                in_reply_to=row.get("in_reply_to", "").strip() or None,
            )
        except Exception as exc:
            print("Failed to parse row id=%r: %s", row.get("id"), exc)
            return None

    def _save_cache(self, tweets: list[Tweet], cache_path: Path) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [t.to_dict() for t in tweets]
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("Saved %d tweets to cache: %s", len(tweets), cache_path)

    def _load_cache(self, cache_path: Path) -> list[Tweet]:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return [Tweet.from_dict(item) for item in payload]
