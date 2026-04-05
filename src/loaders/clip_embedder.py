from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

if TYPE_CHECKING:
    from .models import ExtractedImage

logger = logging.getLogger(__name__)

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


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

    def embed_extracted_images(self, images: list[ExtractedImage]) -> list[ExtractedImage]:
        """Generate CLIP embeddings in-place for a list of ExtractedImage objects.

        Each image is loaded via ExtractedImage.load_pil(), so this works whether
        the raw bytes are still in memory or only the saved file path is available.
        Returns the same list for convenience.
        """
        pil_images: list[Image.Image] = []
        valid: list[ExtractedImage] = []

        for img in images:
            try:
                pil_images.append(img.load_pil())
                valid.append(img)
            except Exception as e:
                logger.warning(
                    "Could not load image p%d idx%d from %s: %s",
                    img.page, img.image_index, img.source, e,
                )

        if pil_images:
            embeddings = self.embed_images(pil_images)
            for img, emb in zip(valid, embeddings):
                img.embedding = emb

        return images
