"""
DRMERS CLUB Full Scraper
Fetches products, generates embeddings, imports to Supabase.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from supabase import create_client, Client

from config import EMBEDDING_DIM, SUPABASE_KEY, SUPABASE_URL
from embeddings import SigLIPEmbedder
from scraper import fetch_all_products, transform_product

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def build_info_text(product: Dict[str, Any]) -> str:
    """Build concatenated text for info_embedding from all product fields."""
    parts = [
        product.get("title", ""),
        product.get("brand", ""),
        product.get("description", ""),
        product.get("category", ""),
        product.get("gender", ""),
        product.get("price", ""),
        product.get("size", ""),
    ]
    if product.get("metadata"):
        try:
            meta = json.loads(product["metadata"])
            parts.append(str(meta.get("product_type", "")))
            parts.append(" ".join(meta.get("tags", [])))
        except (json.JSONDecodeError, TypeError):
            pass
    return " ".join(filter(None, parts))


def prepare_db_record(
    product: Dict[str, Any],
    image_embedding: Optional[List[float]],
    info_embedding: Optional[List[float]],
) -> Dict[str, Any]:
    """Prepare record for Supabase upsert."""
    record = dict(product)
    record["created_at"] = datetime.utcnow().isoformat() + "Z"

    if image_embedding and len(image_embedding) == EMBEDDING_DIM:
        record["image_embedding"] = image_embedding
    else:
        record["image_embedding"] = None

    if info_embedding and len(info_embedding) == EMBEDDING_DIM:
        record["info_embedding"] = info_embedding
    else:
        record["info_embedding"] = None

    return record


def run(
    skip_embeddings: bool = False,
    limit: Optional[int] = None,
    dry_run: bool = False,
):
    """Run full scrape and import pipeline."""
    logger.info("Starting DRMERS CLUB scraper...")

    # 1. Fetch products
    logger.info("Fetching products from Shopify API...")
    raw_products = fetch_all_products()
    if not raw_products:
        logger.error("No products fetched. Exiting.")
        return

    if limit:
        raw_products = raw_products[:limit]
        logger.info(f"Limited to {limit} products")

    logger.info(f"Processing {len(raw_products)} products...")

    # 2. Load embedder
    embedder = None
    if not skip_embeddings:
        embedder = SigLIPEmbedder()

    # 3. Supabase client (only when not dry-run)
    supabase: Optional[Client] = None
    if not dry_run:
        if not SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_SERVICE_KEY is required. Set it in .env or as environment variable."
            )
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    success = 0
    failed = 0

    for i, raw in enumerate(raw_products):
        try:
            product = transform_product(raw)
            if not product.get("image_url"):
                logger.warning(f"Product {product['id']} has no image, skipping embeddings")
                image_embedding = None
                info_embedding = None
            elif embedder:
                image_embedding = embedder.get_image_embedding(product["image_url"])
                info_text = build_info_text(product)
                info_embedding = embedder.get_text_embedding(info_text)
            else:
                image_embedding = None
                info_embedding = None

            record = prepare_db_record(product, image_embedding, info_embedding)

            if dry_run:
                logger.info(f"[DRY RUN] Would upsert: {product['title'][:50]}...")
                success += 1
                continue

            # Upsert (on conflict: source, product_url)
            assert supabase is not None
            supabase.table("products").upsert(
                record,
                on_conflict="source, product_url",
            ).execute()
            success += 1
            logger.info(f"[{i+1}/{len(raw_products)}] Imported: {product['title'][:50]}...")

        except Exception as e:
            failed += 1
            logger.error(f"Failed product {raw.get('id')}: {e}", exc_info=True)

    logger.info(f"Done. Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of products")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    args = parser.parse_args()

    run(
        skip_embeddings=args.skip_embeddings,
        limit=args.limit,
        dry_run=args.dry_run,
    )
