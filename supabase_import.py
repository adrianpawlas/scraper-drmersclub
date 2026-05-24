"""
Supabase import module.
Handles upserting product data into the Supabase 'products' table.
"""

import logging
import time
from typing import Any

from supabase import Client, create_client

from config import BATCH_SIZE, SUPABASE_KEY, SUPABASE_TABLE, SUPABASE_URL

logger = logging.getLogger(__name__)


class SupabaseImporter:
    """Handles importing product data to Supabase."""

    def __init__(self):
        logger.info("Initializing Supabase client...")
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.table = self.client.table(SUPABASE_TABLE)

        # Test connection
        try:
            resp = self.table.select("id").limit(1).execute()
            logger.info("Supabase connection successful.")
            existing = self.table.select("id").execute()
            if existing.data:
                logger.info(f"Found {len(existing.data)} existing products in table")
            else:
                logger.info("No existing products found (table may be empty)")
        except Exception as e:
            logger.warning(f"Could not verify Supabase table: {e}")

    @staticmethod
    def _prepare_entry(entry: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare a product entry for Supabase insertion by converting
        vector embeddings to the pgvector-compatible string format.
        """
        entry = {k: v for k, v in entry.items() if k not in ("info_text",)}

        # Convert vector embeddings to pgvector string format
        # pgvector accepts JSON array strings like "[0.1, 0.2, ...]"
        for vec_field in ("image_embedding", "info_embedding"):
            val = entry.get(vec_field)
            if isinstance(val, list):
                # Format: "[x.xxxx, y.yyyy, ...]" with high precision
                float_strs = [f"{v:.8f}" for v in val]
                entry[vec_field] = "[" + ",".join(float_strs) + "]"

        return entry

    def import_products_batch(
        self, products: list[dict[str, Any]]
    ) -> tuple[int, int]:
        """
        Import a batch of products into Supabase.

        Returns:
            Tuple of (success_count, failure_count)
        """
        success = 0
        failed = 0

        entries = [self._prepare_entry(p) for p in products]

        try:
            self.table.upsert(entries, on_conflict="id").execute()
            success = len(entries)
            logger.info(f"  Batch import: {success} products upserted successfully")
        except Exception as e:
            logger.error(f"Batch upsert failed: {e}")
            # Fall back to individual imports
            logger.info("  Falling back to individual imports...")
            for entry in entries:
                try:
                    self.table.upsert(entry, on_conflict="id").execute()
                    success += 1
                except Exception as e2:
                    logger.error(
                        f"Error importing {entry.get('id', 'unknown')}: {e2}"
                    )
                    failed += 1

        return success, failed

    def import_all_products(
        self, products: list[dict[str, Any]]
    ) -> tuple[int, int]:
        """
        Import all products in batches.

        Args:
            products: List of product dicts with embeddings

        Returns:
            Tuple of (total_success, total_failed)
        """
        total_success = 0
        total_failed = 0
        total = len(products)

        logger.info(f"Importing {total} products to Supabase in batches of {BATCH_SIZE}...")

        for i in range(0, total, BATCH_SIZE):
            batch = products[i : i + BATCH_SIZE]
            logger.info(
                f"Batch {i // BATCH_SIZE + 1}/{(total + BATCH_SIZE - 1) // BATCH_SIZE}: "
                f"products {i + 1}-{min(i + BATCH_SIZE, total)}"
            )

            success, failed = self.import_products_batch(batch)
            total_success += success
            total_failed += failed

            # Small delay between batches to avoid rate limiting
            if i + BATCH_SIZE < total:
                time.sleep(0.5)

        logger.info(
            f"Import complete: {total_success} succeeded, {total_failed} failed"
        )
        return total_success, total_failed
