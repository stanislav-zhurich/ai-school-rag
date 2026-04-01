import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import kagglehub
import pdfplumber
import pytesseract
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 20      # chars; below this threshold triggers OCR fallback
MIN_IMAGE_BYTES = 5_000   # bytes; skip decorative bullets and tiny icons
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

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


@dataclass
class ExtractedImage:
    """A single image extracted from a PDF page."""

    image_bytes: bytes
    page: int
    image_index: int
    source: str
    embedding: list[float] = field(default_factory=list)
    image_path: str = ""
    metadata: dict = field(default_factory=dict)

    def save(self, output_dir: str | Path) -> Path:
        """Persist the raw image bytes to disk and update image_path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self.source).stem
        filename = f"{stem}_p{self.page}_img{self.image_index}.png"
        dest = output_dir / filename
        dest.write_bytes(self.image_bytes)
        self.image_path = str(dest)
        self.metadata["image_path"] = self.image_path
        return dest



# ---------------------------------------------------------------------------
# CLIP embedder — shared encoder for text and images
# ---------------------------------------------------------------------------

class CLIPEmbedder:
    """
    Wraps the CLIP model to produce 512-dimensional embeddings for both
    text and images in the same vector space.

    A single instance should be shared across PDFLoader and ImageExtractor
    so the model is only loaded into memory once.
    """

    def __init__(self, model_id: str = CLIP_MODEL_ID):
        logger.info("Loading CLIP model: %s", model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.eval()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return L2-normalised CLIP text embeddings."""
        inputs = self.processor(text=texts, return_tensors="pt",
                                padding=True, truncation=True)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.tolist()

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Return L2-normalised CLIP image embeddings."""
        inputs = self.processor(images=images, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.tolist()

    def embed_image(self, image: Image.Image) -> list[float]:
        return self.embed_images([image])[0]


# ---------------------------------------------------------------------------
# Text loader
# ---------------------------------------------------------------------------

class PDFLoader:
    """
    Loads and parses PDF files into structured PageDocument / LoadedDocument
    objects.

    Text extraction strategy per page:
      1. pdfplumber reads the native text layer (fast, accurate).
      2. If extracted text is below MIN_TEXT_LENGTH the page is likely a
         scanned image — pytesseract OCR is used as a fallback.
    """

    def __init__(self, ocr_enabled: bool = True, ocr_dpi: int = 300):
        self.ocr_enabled = ocr_enabled
        self.ocr_dpi = ocr_dpi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, path: str | Path) -> LoadedDocument:
        """Parse a single PDF and return a LoadedDocument."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

        logger.info("Loading %s", path.name)
        pages = self._extract_pages(path)
        return LoadedDocument(source=path.name, pages=pages, total_pages=len(pages))

    def load_files(
        self,
        file_paths: list[str | Path],
        max_files: int | None = None,
    ) -> list[LoadedDocument]:
        """Parse a list of PDF file paths."""
        if max_files:
            file_paths = file_paths[:max_files]
        documents = []
        for pdf_path in file_paths:
            try:
                documents.append(self.load_file(pdf_path))
            except Exception as e:
                logger.warning("Failed to load %s: %s", pdf_path, e)
        return documents

    def load_directory(
        self,
        directory: str | Path,
        glob: str = "**/*.pdf",
        max_files: int | None = None,
    ) -> list[LoadedDocument]:
        """Parse all PDFs found under a directory (recursive by default)."""
        directory = Path(directory)
        pdf_files = sorted(directory.glob(glob))
        if max_files:
            pdf_files = pdf_files[:max_files]
        logger.info("Found %d PDF(s) in %s", len(pdf_files), directory)
        return self.load_files(pdf_files)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_pages(self, path: Path) -> list[PageDocument]:
        pages = []
        with pdfplumber.open(path) as pdf:
            total = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                text, method = self._extract_text_from_page(page, path, page_num)
                pages.append(PageDocument(
                    text=text,
                    metadata={
                        "source": path.name,
                        "page": page_num,
                        "total_pages": total,
                        "content_type": "text",
                        "extraction_method": method,
                    },
                ))
        return pages

    def _extract_text_from_page(
        self,
        page,       # pdfplumber.Page
        path: Path,
        page_num: int,
    ) -> tuple[str, str]:
        """Return (text, method) where method is 'pdfplumber' or 'ocr'."""
        text = page.extract_text() or ""
        if len(text.strip()) >= MIN_TEXT_LENGTH:
            return text.strip(), "pdfplumber"

        if not self.ocr_enabled:
            logger.debug(
                "Page %d of %s has no text layer and OCR is disabled",
                page_num, path.name,
            )
            return text.strip(), "pdfplumber"

        logger.debug(
            "Page %d of %s is image-based, falling back to OCR",
            page_num, path.name,
        )
        return self._ocr_page(page, path, page_num), "ocr"

    def _ocr_page(self, page, path: Path, page_num: int) -> str:
        """Render a PDF page to a PIL image and run Tesseract OCR."""
        try:
            img: Image.Image = page.to_image(resolution=self.ocr_dpi).original
            return pytesseract.image_to_string(img).strip()
        except Exception as e:
            logger.warning("OCR failed on page %d of %s: %s", page_num, path.name, e)
            return ""


# ---------------------------------------------------------------------------
# Image extractor
# ---------------------------------------------------------------------------

class ImageExtractor:
    """
    Extracts embedded images from PDF pages using PyMuPDF (fitz) and
    generates CLIP embeddings so they can be stored in and retrieved from
    the same ChromaDB collection as text chunks.

    Usage
    -----
    embedder = CLIPEmbedder()
    extractor = ImageExtractor(embedder)

    images = extractor.extract_and_embed("data/raw/agricola.pdf",
                                          image_dir="data/processed/images")
    # images is a list[ExtractedImage] ready for ChromaDB upsert
    """

    def __init__(self, embedder: CLIPEmbedder):
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_and_embed(
        self,
        pdf_path: str | Path,
        image_dir: str | Path = "data/processed/images",
    ) -> list[ExtractedImage]:
        """Extract images, save to disk, generate CLIP embeddings, return all."""
        images = self._extract_raw(pdf_path)
        if not images:
            return []
        self._embed(images)
        for img in images:
            img.save(image_dir)
        logger.info(
            "Extracted and embedded %d image(s) from %s",
            len(images), Path(pdf_path).name,
        )
        return images

    def extract_from_files(
        self,
        pdf_paths: list[str | Path],
        image_dir: str | Path = "data/processed/images",
    ) -> list[ExtractedImage]:
        """Process a list of PDF files and return all extracted images."""
        all_images: list[ExtractedImage] = []
        for path in pdf_paths:
            try:
                all_images.extend(self.extract_and_embed(path, image_dir))
            except Exception as e:
                logger.warning("Image extraction failed for %s: %s", path, e)
        return all_images

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_raw(self, pdf_path: str | Path) -> list[ExtractedImage]:
        """Use PyMuPDF to pull raw image bytes from every page."""
        pdf_path = Path(pdf_path)
        images: list[ExtractedImage] = []

        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc, start=1):
            for img_index, img_ref in enumerate(page.get_images(full=True)):
                xref = img_ref[0]
                raw = doc.extract_image(xref)
                image_bytes: bytes = raw["image"]

                if len(image_bytes) < MIN_IMAGE_BYTES:
                    continue  # skip tiny decorative elements

                images.append(ExtractedImage(
                    image_bytes=image_bytes,
                    page=page_num,
                    image_index=img_index,
                    source=pdf_path.name,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num,
                        "content_type": "image",
                        "extraction_method": "pymupdf",
                    },
                ))

        doc.close()
        return images

    def _embed(self, images: list[ExtractedImage]) -> None:
        """Generate CLIP embeddings in-place for a list of ExtractedImage objects."""
        pil_images: list[Image.Image] = []
        valid: list[ExtractedImage] = []

        for img in images:
            try:
                import io
                pil_images.append(Image.open(io.BytesIO(img.image_bytes)).convert("RGB"))
                valid.append(img)
            except Exception as e:
                logger.warning(
                    "Could not decode image p%d idx%d from %s: %s",
                    img.page, img.image_index, img.source, e,
                )

        if not pil_images:
            return

        embeddings = self.embedder.embed_images(pil_images)
        for img, emb in zip(valid, embeddings):
            img.embedding = emb
