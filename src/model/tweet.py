import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Tweet:
    """A single post from the DJT tweets dataset."""

    id: str
    date: datetime
    platform: str
    handle: str
    text: str
    favorite_count: int
    repost_count: int
    quote_flag: bool
    repost_flag: bool
    deleted_flag: bool
    word_count: int
    hashtags: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    user_mentions: list[str] = field(default_factory=list)
    media_count: int = 0
    media_urls: list[str] = field(default_factory=list)
    post_url: str = ""
    in_reply_to: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def year(self) -> int:
        return self.date.year

    @property
    def month(self) -> str:
        return self.date.strftime("%Y-%m")

    def to_chunk_text(self) -> str:
        """Return a formatted string suitable for embedding and RAG retrieval."""
        parts = [
            f"[{self.date.strftime('%Y-%m-%d')} | {self.platform}]",
            self.text,
        ]
        if self.hashtags:
            parts.append("Hashtags: " + " ".join(f"#{h}" for h in self.hashtags))
        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        d = asdict(self)
        d["date"] = self.date.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Tweet":
        """Deserialise from a dict produced by `to_dict`."""
        data = dict(data)
        data["date"] = datetime.fromisoformat(data["date"])
        return cls(**data)
