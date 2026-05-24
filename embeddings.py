"""
Embedding module for generating image and text embeddings using
google/siglip-base-patch16-384 from HuggingFace.

Both image_embedding and info_embedding are 768-dimensional vectors.
"""

import io
import logging
import time
from typing import Any

import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from config import (
    DOWNLOAD_TIMEOUT,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_ID,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    RETRY_DELAY,
)

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates image and text embeddings using SigLIP model."""

    def __init__(self, device: str | None = None):
        """
        Initialize the model and processor.

        Args:
            device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"Loading {EMBEDDING_MODEL_ID} on {device}...")

        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL_ID).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_ID)

        logger.info(f"Model loaded. Embedding dimension: {EMBEDDING_DIM}")

    @torch.no_grad()
    def get_image_embedding(self, image_url: str) -> list[float] | None:
        """
        Download an image from URL and generate its embedding.
        Uses SigLIP's get_image_features() -> pooler_output for 768-dim embedding.
        """
        image = self._download_image(image_url)
        if image is None:
            return None

        try:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)

            outputs = self.model.get_image_features(pixel_values, return_dict=True)
            embedding = outputs.pooler_output  # [1, 768]

            # L2 normalize
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
            return embedding[0].cpu().tolist()
        except Exception as e:
            logger.error(f"Error generating image embedding for {image_url}: {e}")
            return None

    @torch.no_grad()
    def get_text_embedding(self, text: str) -> list[float] | None:
        """
        Generate a text embedding for the given text.
        Uses SigLIP's get_text_features() -> pooler_output for 768-dim embedding.
        """
        if not text or not text.strip():
            return None

        try:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(self.device)

            outputs = self.model.get_text_features(
                inputs.input_ids, inputs.get("attention_mask"), return_dict=True
            )
            embedding = outputs.pooler_output  # [1, 768]

            # L2 normalize
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
            return embedding[0].cpu().tolist()
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return None

    def _download_image(self, url: str) -> Image.Image | None:
        """Download an image from URL, with retry logic."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content))
                # Convert to RGB if necessary (SigLIP expects RGB)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            except Exception as e:
                last_error = e
                logger.warning(f"Image download failed (attempt {attempt + 1}/{MAX_RETRIES}): {url} - {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
        logger.error(f"Failed to download image after {MAX_RETRIES} attempts: {url}")
        return None

    def embed_product(self, product: dict[str, Any]) -> dict[str, Any]:
        """
        Generate both image and text embeddings for a product dict.
        Modifies and returns the product dict in-place.

        The product dict should have 'image_url' and 'info_text' keys.
        """
        result = dict(product)

        # Image embedding
        image_url = result.get("image_url")
        if image_url:
            logger.info(f"  Generating image embedding for {result.get('title', '')}")
            result["image_embedding"] = self.get_image_embedding(image_url)
        else:
            result["image_embedding"] = None

        # Text embedding
        info_text = result.pop("info_text", "")
        if info_text:
            logger.info(f"  Generating text embedding for {result.get('title', '')}")
            text_emb = self.get_text_embedding(info_text)
            result["info_embedding"] = text_emb
        else:
            result["info_embedding"] = None

        return result

    def embed_products_batch(
        self, products: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate embeddings for a batch of products."""
        results = []
        for i, product in enumerate(products):
            try:
                embedded = self.embed_product(product)
                results.append(embedded)
                if (i + 1) % 10 == 0:
                    logger.info(f"  Embedded {i + 1}/{len(products)} products")
            except Exception as e:
                logger.error(
                    f"Error embedding product {product.get('title', 'unknown')}: {e}"
                )
                results.append(product)
        return results
