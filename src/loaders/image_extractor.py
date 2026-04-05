import logging
from pathlib import Path

import fitz  # PyMuPDF

from model.models import ExtractedImage

logger = logging.getLogger(__name__)

MIN_IMAGE_BYTES = 5_000  # bytes; skip decorative bullets and tiny icons

_DEFAULT_IMAGE_DIR = Path("data/processed/images")


class ImageExtractor:
    """
    Extracts embedded images from PDF pages and saves them to disk.

    Embedding is intentionally excluded — call CLIPEmbedder.embed_extracted_images()
    as a separate step so extraction and embedding can be run independently.

    If *processed_dir* is provided the extractor writes a per-PDF manifest JSON
    alongside the PNG files and uses it as a cache on subsequent runs, so PDFs
    are never re-parsed unnecessarily.

    Usage
    -----
    extractor = ImageExtractor(processed_dir="data/processed/images")
    images = extractor.extract_from_files(pdf_paths)

    embedder = CLIPEmbedder()
    embedder.embed_extracted_images(images)
    """

    def __init__(self, processed_dir: str | Path | None = None):
        self.processed_dir = Path(processed_dir) if processed_dir else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, pdf_paths: list[str | Path]) -> list[ExtractedImage]:
        """Extract and save images from a list of PDF files."""
        all_images: list[ExtractedImage] = []
        for path in pdf_paths:
            try:
                all_images.extend(self._process_file(path))
            except Exception as e:
                logger.warning("Image extraction failed for %s: %s", path, e)
        return all_images

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _process_file(self, pdf_path: str | Path) -> list[ExtractedImage]:
        """Orchestrate the extract → save pipeline for a single PDF."""
        pdf_path = Path(pdf_path)

        if self._is_cached(pdf_path):
            return self._load_cached(pdf_path)

        images = self._extract(pdf_path)
        self._save(images, pdf_path)
        return images

    def _extract(self, pdf_path: Path) -> list[ExtractedImage]:
        """Step 1 — pull raw image bytes from every page of the PDF."""
        images: list[ExtractedImage] = []

        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc, start=1):
            for img_index, img_ref in enumerate(page.get_images(full=True)):
                xref = img_ref[0]
                image_bytes: bytes = doc.extract_image(xref)["image"]

                if len(image_bytes) < MIN_IMAGE_BYTES:
                    continue  # skip tiny decorative elements

                images.append(ExtractedImage(
                    page=page_num,
                    image_index=img_index,
                    source=pdf_path.name,
                    image_bytes=image_bytes,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num,
                        "content_type": "image",
                        "extraction_method": "pymupdf",
                    },
                ))
        doc.close()

        logger.info("Extracted %d image(s) from %s", len(images), pdf_path.name)
        return images

    def _save(self, images: list[ExtractedImage], pdf_path: Path) -> None:
        """Step 2 — write PNG files to disk and persist the manifest cache."""
        output_dir = self.processed_dir or _DEFAULT_IMAGE_DIR

        for img in images:
            img.save(output_dir)

        if self.processed_dir is not None:
            manifest_path = self.processed_dir / (pdf_path.stem + "_manifest.json")
            ExtractedImage.save_manifest(images, manifest_path)
            logger.debug("Saved manifest to %s", manifest_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_cached(self, pdf_path: Path) -> bool:
        if self.processed_dir is None:
            return False
        return (self.processed_dir / (pdf_path.stem + "_manifest.json")).exists()

    def _load_cached(self, pdf_path: Path) -> list[ExtractedImage]:
        manifest_path = self.processed_dir / (pdf_path.stem + "_manifest.json")
        logger.info("Loading cached images for %s from %s", pdf_path.name, manifest_path)
        return ExtractedImage.load_manifest(manifest_path)
