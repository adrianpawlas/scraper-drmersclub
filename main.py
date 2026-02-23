"""
DRMERS CLUB Full Scraper
Fetches products, generates embeddings, imports to Supabase.
Uses PostgREST HTTP for reliable CI imports and smart sync (no overwrite, delete stale).
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import EMBEDDING_DIM, SOURCE, SUPABASE_KEY, SUPABASE_URL
from db import SupabaseREST
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

    # 3. Process all products (transform, embed, prepare records)
    records: List[Dict[str, Any]] = []
    process_failed = 0

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
            records.append(record)

            if dry_run:
                logger.info(f"[DRY RUN] Would import: {product['title'][:50]}...")

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(raw_products)} products...")

        except Exception as e:
            process_failed += 1
            logger.error(f"Failed product {raw.get('id')}: {e}", exc_info=True)

    logger.info(f"Processed {len(records)} products ({process_failed} failed during processing)")

    if dry_run:
        logger.info(f"[DRY RUN] Would import {len(records)} products, skip DB writes")
        return

    if not records:
        logger.error("No products to import. Exiting with error.")
        sys.exit(1)

    # 4. Smart sync: insert new only (no overwrite), then delete stale
    if not SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_SERVICE_KEY is required. Set it in .env or as environment variable."
        )
    db = SupabaseREST(SUPABASE_URL, SUPABASE_KEY)

    # Insert new products only; existing ones stay untouched (resolution=ignore-duplicates)
    inserted, insert_failed = db.upsert_new_only(records)
    logger.info(f"Import: {inserted} products written, {insert_failed} failed")

    # Remove products no longer in catalog (stale)
    keep_ids = [r["id"] for r in records]
    deleted, delete_errors = db.delete_stale_products(SOURCE, keep_ids)
    if deleted:
        logger.info(f"Removed {deleted} stale products (no longer in catalog)")
    if delete_errors:
        logger.warning(f"Delete had {delete_errors} errors")

    total_success = inserted
    total_failed = process_failed + insert_failed
    logger.info(f"Done. Success: {total_success}, Failed: {total_failed}")

    if total_success == 0:
        logger.error("No products were imported. Exiting with error.")
        sys.exit(1)


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
