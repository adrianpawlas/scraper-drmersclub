#!/usr/bin/env python3
"""
Drmers Club Product Scraper - Main Orchestrator

Pipeline:
1. Fetch all products from Drmers Club via Shopify API
2. Classify against existing database records (new / changed / unchanged / stale)
3. Generate embeddings only for products that need them (new + image changed)
4. Smart upsert to Supabase with batch of 50, stale cleanup, and run summary

Usage:
    python main.py                    # Full smart pipeline
    python main.py --fetch-only       # Only fetch & process products
    python main.py --embed-only       # Only generate embeddings from saved products
    python main.py --import-only      # Only run import from saved products
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


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_products(products: list[dict], filepath: Path = PRODUCTS_FILE) -> None:
    """Save products list to JSON file (checkpoint), stripping embeddings."""
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


# ---------------------------------------------------------------------------
# Step 1 — Fetch
# ---------------------------------------------------------------------------

def step_fetch(limit: int | None = None) -> list[dict]:
    """Fetch and process all products from the store."""
    from scraper import fetch_and_process_all_products

    logger.info("=" * 60)
    logger.info("STEP 1: Fetching products from Drmers Club")
    logger.info("=" * 60)

    products = fetch_and_process_all_products()

    if limit and limit < len(products):
        logger.info(f"Limiting to {limit} products as requested")
        products = products[:limit]

    products = add_created_at(products)
    save_products(products)

    logger.info(f"Step 1 complete: {len(products)} products fetched and processed")
    return products


# ---------------------------------------------------------------------------
# Step 2 — Embed (smart: only new + image-changed products)
# ---------------------------------------------------------------------------

def step_embed(products: list[dict] | None = None) -> list[dict]:
    """Generate embeddings for products that need them (new or image changed)."""
    from embeddings import EmbeddingGenerator

    if products is None:
        products = load_products()

    if not products:
        logger.error("No products to embed.")
        return []

    # Determine which products need embeddings
    needs_embed = [p for p in products if not p.get("image_embedding")]
    skip_count = len(products) - len(needs_embed)

    logger.info("=" * 60)
    logger.info(f"STEP 2: Generating embeddings for {len(needs_embed)}/{len(products)} products")
    logger.info("=" * 60)

    if skip_count > 0:
        logger.info(f"  {skip_count} products already have embeddings — skipping")

    if not needs_embed:
        logger.info("No products need embeddings — all already have them.")
        return products

    generator = EmbeddingGenerator()
    embedded_results = generator.embed_products_batch(needs_embed)

    # Merge embedded products back into the full list
    embedded_by_id: dict[str, dict] = {}
    for p in embedded_results:
        pid = p.get("id") or p.get("product_url", "")
        embedded_by_id[pid] = p

    final_products = []
    for p in products:
        pid = p.get("id") or p.get("product_url", "")
        if pid in embedded_by_id:
            final_products.append(embedded_by_id[pid])
        else:
            final_products.append(p)

    # Save checkpoint
    save_products(final_products, EMBEDDED_FILE)

    image_ok = sum(1 for p in final_products if p.get("image_embedding"))
    text_ok = sum(1 for p in final_products if p.get("info_embedding"))
    logger.info(
        f"Step 2 complete: {image_ok} image embeddings, {text_ok} text embeddings"
    )

    return final_products


# ---------------------------------------------------------------------------
# Step 3 — Smart Import
# ---------------------------------------------------------------------------

def step_import(products: list[dict] | None = None) -> dict[str, int]:
    """
    Run the smart import pipeline:
      - Classify products against existing DB records
      - Generate embeddings only for new / image-changed
      - Batch upsert (50/batch) with retry logic
      - Mark/delete stale products
      - Print run summary

    Returns stats dict.
    """
    from supabase_import import SupabaseImporter

    if products is None:
        products = load_products(EMBEDDED_FILE)
        if not products:
            products = load_products()

    if not products:
        logger.error("No products to import.")
        return {"new": 0, "updated": 0, "unchanged": 0, "stale_deleted": 0, "failed": 0}

    logger.info("=" * 60)
    logger.info(f"STEP 3: Smart import — {len(products)} scraped products")
    logger.info("=" * 60)

    # Initialize importer (loads existing products from DB)
    importer = SupabaseImporter()

    # Classify: separate into new / changed / unchanged / stale
    classification = importer.classify_products(products)

    # Determine which products need new embeddings
    needs_embed = importer.products_needing_embeddings(classification)

    # Generate embeddings for products that need them
    if needs_embed:
        logger.info(f"Generating embeddings for {len(needs_embed)} products...")
        from embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator()
        embedded = generator.embed_products_batch(needs_embed)

        # Merge embeddings back into the correct category lists
        embed_lookup: dict[str, dict] = {}
        for p in embedded:
            pid = p.get("id") or p.get("product_url", "")
            embed_lookup[pid] = p

        def _merge_embeddings(category_list: list[dict]) -> list[dict]:
            merged = []
            for p in category_list:
                pid = p.get("id") or p.get("product_url", "")
                if pid in embed_lookup:
                    merged.append(embed_lookup[pid])
                else:
                    merged.append(p)
            return merged

        classification["new"] = _merge_embeddings(classification["new"])
        classification["changed"] = _merge_embeddings(classification["changed"])

    # Run the import (upsert new+changed, mark/delete stale, print summary)
    stats = importer.run_import(products)

    # Save final checkpoint
    save_products(products, EMBEDDED_FILE)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Drmers Club Product Scraper — Smart Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py                    # Full smart pipeline
  python main.py --limit 5          # Test with just 5 products
  python main.py --fetch-only       # Only fetch products
  python main.py --embed-only       # Only generate embeddings
  python main.py --import-only      # Only run smart import (uses existing + DB)
  python main.py --skip-embedding   # Fetch + import (no embeddings)
        """,
    )
    parser.add_argument("--fetch-only", action="store_true",
                        help="Only fetch products (no embeddings/import)")
    parser.add_argument("--embed-only", action="store_true",
                        help="Only generate embeddings from saved products")
    parser.add_argument("--import-only", action="store_true",
                        help="Only run smart import (fetch + gen + classify + upsert + stale cleanup)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of products to process")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Skip embedding generation (fetch + import only)")

    args = parser.parse_args()

    start_time = time.time()

    try:
        if args.fetch_only:
            products = step_fetch(args.limit)
            logger.info(f"Fetch complete. {len(products)} products saved to {PRODUCTS_FILE}")

        elif args.embed_only:
            products = load_products()
            if args.limit:
                products = products[:args.limit]
            step_embed(products)

        elif args.import_only:
            # Smart import: loads saved products, classifies against DB, generates
            # needed embeddings, upserts, cleans stale, prints summary
            products = load_products(EMBEDDED_FILE)
            if not products:
                products = load_products()
            if args.limit:
                products = products[:args.limit]
            step_import(products)

        elif args.resume:
            products = load_products(EMBEDDED_FILE)
            if products:
                logger.info("Found embedded products, running smart import...")
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
                    # Full smart pipeline: classify -> embed needed -> import
                    from supabase_import import SupabaseImporter
                    importer = SupabaseImporter()
                    classification = importer.classify_products(products)
                    needs_embed = importer.products_needing_embeddings(classification)
                    if needs_embed:
                        embedded = step_embed(needs_embed)
                        # Merge back
                        embed_lookup = {p.get("product_url", ""): p for p in embedded}
                        for cat in ("new", "changed"):
                            classification[cat] = [
                                embed_lookup.get(p.get("product_url", ""), p)
                                for p in classification[cat]
                            ]
                    importer.run_import(products)

        elif args.skip_embedding:
            products = step_fetch(args.limit)
            step_import(products)

        else:
            # Full pipeline
            products = step_fetch(args.limit)
            step_import(products)

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
