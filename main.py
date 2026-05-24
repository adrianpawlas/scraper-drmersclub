#!/usr/bin/env python3
"""
Drmers Club Product Scraper - Main Orchestrator

This script:
1. Fetches all products from Drmers Club via Shopify API
2. Generates image and text embeddings using google/siglip-base-patch16-384
3. Imports everything to Supabase products table

Usage:
    python main.py                    # Full pipeline
    python main.py --fetch-only       # Only fetch & process products (no embeddings/import)
    python main.py --embed-only       # Only generate embeddings from saved products
    python main.py --import-only      # Only import from saved products with embeddings
    python main.py --resume           # Resume from last checkpoint
    python main.py --limit N          # Process only N products (for testing)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("drmers-scraper")

# Data directory for checkpoints
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PRODUCTS_FILE = DATA_DIR / "products.json"
EMBEDDED_FILE = DATA_DIR / "products_embedded.json"


def save_products(products: list[dict], filepath: Path = PRODUCTS_FILE) -> None:
    """Save products list to JSON file (checkpoint)."""
    # Remove embeddings and temp fields from saved file to save space
    save_data = []
    for p in products:
        entry = dict(p)
        entry.pop("image_embedding", None)
        entry.pop("info_embedding", None)
        entry.pop("info_text", None)
        save_data.append(entry)

    with open(filepath, "w") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(save_data)} products to {filepath}")


def load_products(filepath: Path = PRODUCTS_FILE) -> list[dict]:
    """Load products from JSON checkpoint file."""
    if not filepath.exists():
        logger.error(f"No checkpoint file found at {filepath}")
        return []
    with open(filepath, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} products from {filepath}")
    return data


def add_created_at(products: list[dict]) -> list[dict]:
    """Add created_at timestamp to each product."""
    now = datetime.now(timezone.utc).isoformat()
    for p in products:
        p["created_at"] = now
    return products


def step_fetch(limit: int | None = None) -> list[dict]:
    """Step 1: Fetch and process all products from the store."""
    from scraper import fetch_and_process_all_products

    logger.info("=" * 60)
    logger.info("STEP 1: Fetching products from Drmers Club")
    logger.info("=" * 60)

    products = fetch_and_process_all_products()

    if limit and limit < len(products):
        logger.info(f"Limiting to {limit} products as requested")
        products = products[:limit]

    # Add created_at timestamp
    products = add_created_at(products)

    # Save checkpoint
    save_products(products)

    logger.info(f"Step 1 complete: {len(products)} products fetched and processed")
    return products


def step_embed(products: list[dict] | None = None) -> list[dict]:
    """Step 2: Generate embeddings for all products."""
    from embeddings import EmbeddingGenerator

    if products is None:
        products = load_products()

    if not products:
        logger.error("No products to embed. Run --fetch-only first or provide products.")
        return []

    logger.info("=" * 60)
    logger.info(f"STEP 2: Generating embeddings for {len(products)} products")
    logger.info("=" * 60)

    # Initialize the model (this downloads it the first time)
    generator = EmbeddingGenerator()

    # Process products
    embedded_products = generator.embed_products_batch(products)

    # Save embedded checkpoint
    save_products(embedded_products, EMBEDDED_FILE)

    # Count successes
    image_ok = sum(1 for p in embedded_products if p.get("image_embedding"))
    text_ok = sum(1 for p in embedded_products if p.get("info_embedding"))
    logger.info(
        f"Step 2 complete: {image_ok} image embeddings, {text_ok} text embeddings generated"
    )

    return embedded_products


def step_import(products: list[dict] | None = None) -> None:
    """Step 3: Import all products to Supabase."""
    from supabase_import import SupabaseImporter

    if products is None:
        # Try embedded file first, then fall back to regular products
        products = load_products(EMBEDDED_FILE)
        if not products:
            products = load_products()

    if not products:
        logger.error("No products to import.")
        return

    logger.info("=" * 60)
    logger.info(f"STEP 3: Importing {len(products)} products to Supabase")
    logger.info("=" * 60)

    importer = SupabaseImporter()
    success, failed = importer.import_all_products(products)

    logger.info(f"Step 3 complete: {success} imported, {failed} failed")

    if failed > 0:
        logger.warning(f"{failed} products failed to import. Check logs for details.")


def main():
    parser = argparse.ArgumentParser(
        description="Drmers Club Product Scraper - Full pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline
  python main.py --limit 5          # Test with just 5 products
  python main.py --fetch-only       # Only fetch products
  python main.py --embed-only       # Only generate embeddings
  python main.py --import-only      # Only import to Supabase
        """,
    )
    parser.add_argument(
        "--fetch-only", action="store_true", help="Only fetch products (no embeddings/import)"
    )
    parser.add_argument(
        "--embed-only", action="store_true", help="Only generate embeddings from saved products"
    )
    parser.add_argument(
        "--import-only", action="store_true", help="Only import saved products to Supabase"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of products to process"
    )
    parser.add_argument(
        "--skip-embedding", action="store_true", help="Skip embedding generation (fetch + import only)"
    )

    args = parser.parse_args()

    start_time = time.time()

    try:
        if args.fetch_only:
            # Only fetch
            products = step_fetch(args.limit)
            logger.info(f"Fetch complete. {len(products)} products saved to {PRODUCTS_FILE}")

        elif args.embed_only:
            # Only embed
            products = load_products()
            if args.limit:
                products = products[:args.limit]
            step_embed(products)

        elif args.import_only:
            # Only import
            step_import()

        elif args.resume:
            # Resume: check what checkpoints exist
            products = load_products(EMBEDDED_FILE)
            if products:
                logger.info("Found embedded products, importing to Supabase...")
                step_import(products)
            else:
                products = load_products()
                if products:
                    logger.info("Found fetched products, generating embeddings...")
                    embedded = step_embed(products)
                    step_import(embedded)
                else:
                    logger.info("No checkpoints found, running full pipeline...")
                    products = step_fetch(args.limit)
                    embedded = step_embed(products)
                    step_import(embedded)

        elif args.skip_embedding:
            # Fetch + import (skip embeddings)
            products = step_fetch(args.limit)
            step_import(products)

        else:
            # Full pipeline
            products = step_fetch(args.limit)
            embedded = step_embed(products)
            step_import(embedded)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Progress has been saved to checkpoint files.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info(f"Total time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
