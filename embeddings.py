"""Image and text embeddings using SigLIP model."""

import io
import logging
import time
from typing import List, Optional

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, SiglipModel

from config import EMBEDDING_DIM, SIGLIP_MODEL

logger = logging.getLogger(__name__)


class SigLIPEmbedder:
    """SigLIP model for 768-dim image and text embeddings."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SigLIP model {SIGLIP_MODEL} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL)
        self.model = SiglipModel.from_pretrained(SIGLIP_MODEL).to(self.device)
        self.model.eval()
        logger.info(f"Model loaded. Output dim must match Supabase vector column ({EMBEDDING_DIM}).")

    def _get_image_from_url(self, url: str, retries: int = 2) -> Optional[Image.Image]:
        """Download image from URL and return PIL Image."""
        for attempt in range(retries + 1):
            try:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                if attempt == retries:
                    logger.warning(f"Failed to load image {url}: {e}")
                    return None
                time.sleep(1)
        return None

    def get_image_embedding(self, image_url: str) -> Optional[List[float]]:
        """Get 768-dim embedding for product image."""
        image = self._get_image_from_url(image_url)
        if image is None:
            return None

        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            # Handle tensor or model output object
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state[:, 0]  # CLS token
            elif hasattr(outputs, "__getitem__"):
                emb = outputs[0]
            else:
                emb = outputs
            embedding = emb.cpu().float().numpy().squeeze()
            # Ensure correct dimension
            if len(embedding) != EMBEDDING_DIM:
                logger.warning(
                    f"SigLIP output dim {len(embedding)} != expected {EMBEDDING_DIM}"
                )
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"Embedding failed for {image_url}: {e}")
            return None

    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """Get 768-dim embedding for product info text using SigLIP text encoder."""
        if not text or not text.strip():
            return None

        try:
            # SigLIP text encoder - use tokenizer (text has max ~64 tokens)
            text_inputs = self.tokenizer(
                text[:500],
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.get_text_features(**text_inputs)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state[:, 0]
            elif hasattr(outputs, "__getitem__"):
                emb = outputs[0]
            else:
                emb = outputs
            embedding = emb.cpu().float().numpy().squeeze()
            if len(embedding) != EMBEDDING_DIM:
                logger.warning(
                    f"SigLIP text output dim {len(embedding)} != expected {EMBEDDING_DIM}"
                )
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"Text embedding failed: {e}")
            return None
