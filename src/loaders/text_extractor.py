import logging
from pathlib import Path

import pdfplumber
import pytesseract
from PIL import Image

from model.models import LoadedDocument, PageDocument

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 20  # chars; below this threshold triggers OCR fallback


class TextExtractor:
    """
    Loads and parses PDF files into structured PageDocument / LoadedDocument
    objects.

    Text extraction strategy per page:
      1. pdfplumber reads the native text layer (fast, accurate).
      2. If extracted text is below MIN_TEXT_LENGTH the page is likely a
         scanned image — pytesseract OCR is used as a fallback.

    If *processed_dir* is provided the result is persisted as a JSON file
    after extraction and used as a cache on subsequent runs, so PDFs are
    never re-parsed unnecessarily.

    Usage
    -----
    loader = PDFLoader(processed_dir="data/processed/text")
    docs = loader.load_files(pdf_paths)
    """

    def __init__(
        self,
        ocr_enabled: bool = True,
        ocr_dpi: int = 300,
        processed_dir: str | Path | None = None,
    ):
        self.ocr_enabled = ocr_enabled
        self.ocr_dpi = ocr_dpi
        self.processed_dir = Path(processed_dir) if processed_dir else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, path: str | Path) -> LoadedDocument:
        """Parse a single PDF and return a LoadedDocument.

        If *processed_dir* was supplied the cached JSON is returned on
        subsequent calls, skipping PDF parsing and OCR entirely.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

        if self.processed_dir is not None:
            cache_path = self.processed_dir / (path.stem + ".json")
            if cache_path.exists():
                logger.info("Loading cached text for %s from %s", path.name, cache_path)
                return LoadedDocument.load(cache_path)

        logger.info("Loading %s", path.name)
        pages = self._extract_pages(path)
        doc = LoadedDocument(source=path.name, pages=pages, total_pages=len(pages))

        if self.processed_dir is not None:
            doc.save(self.processed_dir)

        return doc

    def extract(
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
        return self.extract(pdf_files)

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
