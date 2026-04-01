import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import kagglehub
import pdfplumber
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 20  # chars below this threshold triggers OCR fallback


@dataclass
class PageDocument:
    """A single page extracted from a PDF."""

    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass
class LoadedDocument:
    """All pages extracted from one PDF file."""

    source: str
    pages: list[PageDocument]
    total_pages: int

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if not p.is_empty)

    @property
    def non_empty_pages(self) -> list[PageDocument]:
        return [p for p in self.pages if not p.is_empty]


def download_dataset(kaggle_dataset_handle: str, output_dir: str = "data/raw") -> list[str]:
    """Download the Epstein documents dataset from Kaggle and return PDF file paths."""
    os.makedirs(output_dir, exist_ok=True)
    path = kagglehub.dataset_download(
        handle=kaggle_dataset_handle,
        output_dir=output_dir,
    )
    logger.info("Dataset downloaded to: %s", path)
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(".pdf")
    ]
    logger.info("Found %d PDF file(s)", len(files))
    return files


class PDFLoader:
    """
    Loads and parses PDF files into structured PageDocument/LoadedDocument objects.

    Strategy per page:
      1. pdfplumber extracts the native text layer (fast, accurate).
      2. If the extracted text is below MIN_TEXT_LENGTH the page is likely a
         scanned image — pytesseract OCR is used as a fallback.
    """

    def __init__(self, ocr_enabled: bool = True, ocr_dpi: int = 300):
        self.ocr_enabled = ocr_enabled
        self.ocr_dpi = ocr_dpi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, path: str | Path) -> LoadedDocument:
        """Parse a single PDF file and return a LoadedDocument."""
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
                pages.append(
                    PageDocument(
                        text=text,
                        metadata={
                            "source": path.name,
                            "page": page_num,
                            "total_pages": total,
                            "extraction_method": method,
                        },
                    )
                )
        return pages

    def _extract_text_from_page(
        self,
        page,  # pdfplumber.Page
        path: Path,
        page_num: int,
    ) -> tuple[str, str]:
        """Return (text, method) where method is 'pdfplumber' or 'ocr'."""
        text = page.extract_text() or ""
        if len(text.strip()) >= MIN_TEXT_LENGTH:
            return text.strip(), "pdfplumber"

        if not self.ocr_enabled:
            logger.debug(
                "Page %d of %s has no text layer and OCR is disabled", page_num, path.name
            )
            return text.strip(), "pdfplumber"

        logger.debug(
            "Page %d of %s is image-based, falling back to OCR", page_num, path.name
        )
        ocr_text = self._ocr_page(page, path, page_num)
        return ocr_text, "ocr"

    def _ocr_page(self, page, path: Path, page_num: int) -> str:
        """Render a PDF page to an image and run Tesseract OCR."""
        try:
            img: Image.Image = page.to_image(resolution=self.ocr_dpi).original
            return pytesseract.image_to_string(img).strip()
        except Exception as e:
            logger.warning("OCR failed on page %d of %s: %s", page_num, path.name, e)
            return ""
