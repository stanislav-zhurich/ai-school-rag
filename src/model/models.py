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


@dataclass
class PageDocument:
    """A single text page extracted from a PDF."""

    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass
class LoadedDocument:
    """All text pages extracted from one PDF file."""

    source: str
    pages: list[PageDocument]
    total_pages: int

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if not p.is_empty)

    @property
    def non_empty_pages(self) -> list[PageDocument]:
        return [p for p in self.pages if not p.is_empty]

    def save(self, output_dir: str | Path) -> Path:
        """Persist extracted text and metadata to a JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / (Path(self.source).stem + ".json")
        payload = {
            "source": self.source,
            "total_pages": self.total_pages,
            "pages": [
                {"text": p.text, "metadata": p.metadata}
                for p in self.pages
            ],
        }
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Saved extracted text to %s", dest)
        return dest

    @classmethod
    def load(cls, json_path: str | Path) -> "LoadedDocument":
        """Restore a LoadedDocument from a previously saved JSON file."""
        json_path = Path(json_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        pages = [
            PageDocument(text=p["text"], metadata=p["metadata"])
            for p in payload["pages"]
        ]
        return cls(source=payload["source"], pages=pages, total_pages=payload["total_pages"])


@dataclass
class ExtractedImage:
    """A single image extracted from a PDF page."""

    page: int
    image_index: int
    source: str
    image_bytes: bytes = field(default=b"")
    embedding: list[float] = field(default_factory=list)
    image_path: str = ""
    metadata: dict = field(default_factory=dict)

    def save(self, output_dir: str | Path) -> Path:
        """Persist the raw image bytes to disk and update image_path."""
        if not self.image_bytes:
            raise ValueError("Cannot save: image_bytes is empty.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self.source).stem
        filename = f"{stem}_p{self.page}_img{self.image_index}.png"
        dest = output_dir / filename
        dest.write_bytes(self.image_bytes)
        self.image_path = str(dest)
        self.metadata["image_path"] = self.image_path
        return dest

    def load_pil(self) -> "Image.Image":
        """Return a PIL Image, reading from image_bytes or image_path."""
        from PIL import Image as PILImage
        if self.image_bytes:
            import io
            return PILImage.open(io.BytesIO(self.image_bytes)).convert("RGB")
        if self.image_path:
            return PILImage.open(self.image_path).convert("RGB")
        raise ValueError("No image data available: both image_bytes and image_path are empty.")

    @classmethod
    def save_manifest(cls, images: "list[ExtractedImage]", manifest_path: Path) -> None:
        """Persist image metadata (no bytes) to a JSON manifest file."""
        payload = [
            {
                "page": img.page,
                "image_index": img.image_index,
                "source": img.source,
                "image_path": img.image_path,
                "metadata": img.metadata,
            }
            for img in images
        ]
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Saved image manifest to %s", manifest_path)

    @classmethod
    def load_manifest(cls, manifest_path: Path) -> "list[ExtractedImage]":
        """Restore ExtractedImage list from a manifest (image_bytes will be empty)."""
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [
            cls(
                page=item["page"],
                image_index=item["image_index"],
                source=item["source"],
                image_path=item["image_path"],
                metadata=item["metadata"],
            )
            for item in payload
        ]
