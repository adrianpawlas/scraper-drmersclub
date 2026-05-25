"""
Supabase import module.
Handles smart upserting of product data into the Supabase 'products' table.

Key features:
- Batch inserts (50 products per batch)
- Smart upsert: classifies products as new / changed / unchanged / stale
- Only updates when data has actually changed
- Removes stale products after 2 consecutive missed runs
- Retry logic for failed batches
- Run summary at end
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from supabase import Client, create_client

from config import (
    BATCH_SIZE,
    EMBEDDING_DELAY,
    FAILED_BATCHES_LOG,
    MAX_UPSERT_RETRIES,
    SOURCE,
    STALE_RUNS_BEFORE_DELETE,
    SUPABASE_KEY,
    SUPABASE_TABLE,
    SUPABASE_URL,
)

logger = logging.getLogger(__name__)

# Fields to compare when determining if a product has changed
COMPARISON_FIELDS = [
    "price",
    "sale",
    "title",
    "description",
    "image_url",
    "category",
    "gender",
    "size",
    "tags",
    "additional_images",
    "brand",
]


class SupabaseImporter:
    """Handles importing product data to Supabase with smart upsert logic."""

    def __init__(self):
        logger.info("Initializing Supabase client...")
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.table = self.client.table(SUPABASE_TABLE)

        # Lookup: existing products by (source, product_url)
        # Stored as {product_url: db_record}
        self.existing_by_url: dict[str, dict[str, Any]] = {}
        # Lookup: existing products by id (for upsert reference)
        self.existing_by_id: dict[str, dict[str, Any]] = {}

        # Fetch existing products for our source
        self._fetch_existing()

        logger.info(
            f"Supabase ready. {len(self.existing_by_url)} existing products loaded."
        )

    def _fetch_existing(self) -> None:
        """Fetch all existing products for this source from the database."""
        try:
            # Fetch in pages (Supabase REST has limits)
            offset = 0
            page_size = 1000
            all_products = []

            while True:
                resp = (
                    self.table.select("*")
                    .eq("source", SOURCE)
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
                batch = resp.data if resp.data else []
                all_products.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size

            # Build lookup dicts
            for record in all_products:
                url = record.get("product_url")
                pid = record.get("id")
                if url:
                    self.existing_by_url[url] = record
                if pid:
                    self.existing_by_id[pid] = record

            logger.info(f"Fetched {len(all_products)} existing products for source '{SOURCE}'.")

        except Exception as e:
            logger.warning(f"Could not fetch existing products: {e}. "
                           "All products will be treated as new.")

    @staticmethod
    def _parse_metadata(record: dict[str, Any]) -> dict[str, Any]:
        """Parse the metadata JSON field from a database record."""
        meta = record.get("metadata")
        if isinstance(meta, str):
            try:
                return json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                return {}
        elif isinstance(meta, dict):
            return meta
        return {}

    @staticmethod
    def _get_missed_count(record: dict[str, Any]) -> int:
        """Get the missed_scrapes counter from a product's metadata."""
        meta = SupabaseImporter._parse_metadata(record)
        return int(meta.get("missed_scrapes", 0))

    @staticmethod
    def _set_missed_count(record: dict[str, Any], count: int) -> str:
        """
        Set the missed_scrapes counter in a product's metadata.
        Returns the updated metadata JSON string.
        """
        meta = SupabaseImporter._parse_metadata(record)
        meta["missed_scrapes"] = count
        # Preserve other metadata fields
        return json.dumps(meta, ensure_ascii=False)

    @staticmethod
    def _has_changed(scraped: dict[str, Any], existing: dict[str, Any]) -> bool:
        """
        Compare a scraped product against the existing database record.
        Returns True if any relevant field has changed.
        """
        for field in COMPARISON_FIELDS:
            scraped_val = str(scraped.get(field, "") or "")
            existing_val = str(existing.get(field, "") or "")
            if scraped_val != existing_val:
                logger.debug(
                    f"  Field '{field}' changed: "
                    f"'{existing_val[:50]}' -> '{scraped_val[:50]}'"
                )
                return True
        return False

    @staticmethod
    def _image_url_changed(scraped: dict[str, Any], existing: dict[str, Any]) -> bool:
        """Check specifically if the image URL has changed (for embedding regeneration)."""
        scraped_img = (scraped.get("image_url") or "").strip()
        existing_img = (existing.get("image_url") or "").strip()
        return scraped_img != existing_img

    def classify_products(
        self, scraped_products: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Compare scraped products against existing database records.

        Returns a dict with keys:
          - 'new': products not in the database (need full embeddings)
          - 'changed': products with changed fields (need new embeddings)
          - 'unchanged': products with no changes (skip embeddings, skip DB update)
          - 'stale_to_delete': existing products not seen for 2+ runs
          - 'stale_to_mark': existing products not seen this run (first missed run)
        """
        seen_urls: set[str] = set()
        new: list[dict[str, Any]] = []
        changed: list[dict[str, Any]] = []
        unchanged: list[dict[str, Any]] = []

        for scraped in scraped_products:
            url = scraped.get("product_url", "")
            seen_urls.add(url)

            existing = self.existing_by_url.get(url)

            if existing is None:
                # Brand new product
                logger.info(f"  NEW: {scraped.get('title', '')}")
                new.append(scraped)
            elif self._has_changed(scraped, existing):
                # Product exists but data has changed
                logger.info(f"  CHANGED: {scraped.get('title', '')}")
                # Preserve the existing id for the upsert
                scraped["id"] = existing["id"]
                changed.append(scraped)
            else:
                # Product unchanged — skip it entirely
                logger.debug(f"  UNCHANGED: {scraped.get('title', '')}")
                unchanged.append(scraped)

        # Detect stale products: existing in DB but not in scrape results
        stale_to_delete: list[dict[str, Any]] = []
        stale_to_mark: list[dict[str, Any]] = []

        for url, existing in self.existing_by_url.items():
            if url not in seen_urls:
                missed_count = self._get_missed_count(existing)
                if missed_count >= (STALE_RUNS_BEFORE_DELETE - 1):
                    # Missed 2+ consecutive runs → delete
                    logger.info(f"  STALE DELETE: {existing.get('title', '')} "
                                f"(missed {missed_count + 1} runs)")
                    stale_to_delete.append(existing)
                else:
                    # First missed run → increment counter
                    new_count = missed_count + 1
                    logger.info(f"  STALE MARK: {existing.get('title', '')} "
                                f"(missed run {new_count}/{STALE_RUNS_BEFORE_DELETE})")
                    # Update the metadata with incremented missed counter
                    updated_meta = self._set_missed_count(existing, new_count)
                    update_entry = {
                        "id": existing["id"],
                        "metadata": updated_meta,
                    }
                    stale_to_mark.append(update_entry)

        logger.info(
            f"Classification: {len(new)} new, {len(changed)} changed, "
            f"{len(unchanged)} unchanged, "
            f"{len(stale_to_delete)} to delete, {len(stale_to_mark)} to mark stale"
        )

        return {
            "new": new,
            "changed": changed,
            "unchanged": unchanged,
            "stale_to_delete": stale_to_delete,
            "stale_to_mark": stale_to_mark,
        }

    def products_needing_embeddings(
        self, classification: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """
        Determine which products need new embeddings.
        Only new products or products whose image URL changed need re-embedding.
        """
        need_embed: list[dict[str, Any]] = []

        # New products always need embeddings
        need_embed.extend(classification["new"])

        # Changed products: only re-embed if image URL changed
        for product in classification["changed"]:
            url = product.get("product_url", "")
            existing = self.existing_by_url.get(url)
            if existing and self._image_url_changed(product, existing):
                logger.info(f"  Re-embedding (image changed): {product.get('title', '')}")
                need_embed.append(product)
            elif not existing:
                # Shouldn't happen for 'changed' products, but safety
                need_embed.append(product)
            else:
                logger.info(f"  Skipping embed (only text/data changed): {product.get('title', '')}")
                # Copy existing embeddings forward
                product["image_embedding"] = existing.get("image_embedding")
                product["info_embedding"] = existing.get("info_embedding")

        # Unchanged products keep their existing embeddings (already carried forward)
        for product in classification["unchanged"]:
            url = product.get("product_url", "")
            existing = self.existing_by_url.get(url)
            if existing:
                product["image_embedding"] = existing.get("image_embedding")
                product["info_embedding"] = existing.get("info_embedding")
                product["id"] = existing["id"]

        logger.info(
            f"Products needing embeddings: {len(need_embed)} "
            f"({len(classification['new'])} new, "
            f"{len(need_embed) - len(classification['new'])} image-changed)"
        )
        return need_embed

    @staticmethod
    def _prepare_entry(entry: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare a product entry for Supabase insertion by converting
        vector embeddings to the pgvector-compatible string format.
        """
        entry = {k: v for k, v in entry.items() if k not in ("info_text",)}

        # Convert vector embeddings to pgvector string format
        for vec_field in ("image_embedding", "info_embedding"):
            val = entry.get(vec_field)
            if isinstance(val, list):
                float_strs = [f"{v:.8f}" for v in val]
                entry[vec_field] = "[" + ",".join(float_strs) + "]"

        return entry

    def _batch_upsert_with_retry(
        self, entries: list[dict[str, Any]], batch_num: int, total_batches: int
    ) -> tuple[int, int]:
        """
        Upsert a single batch with retry logic.

        Returns:
            Tuple of (success_count, failure_count)
        """
        last_error = None

        for attempt in range(1, MAX_UPSERT_RETRIES + 1):
            try:
                self.table.upsert(entries, on_conflict="id").execute()
                return len(entries), 0
            except Exception as e:
                last_error = e
                logger.warning(
                    f"  Batch {batch_num}/{total_batches} failed "
                    f"(attempt {attempt}/{MAX_UPSERT_RETRIES}): {e}"
                )
                if attempt < MAX_UPSERT_RETRIES:
                    time.sleep(1 * attempt)  # Progressive backoff

        # All retries exhausted — log failures and try individual inserts
        logger.error(
            f"  Batch {batch_num}/{total_batches} failed after "
            f"{MAX_UPSERT_RETRIES} attempts. Falling back to individual inserts."
        )

        self._log_failed_batch(entries, last_error)

        success = 0
        failed = 0
        for entry in entries:
            try:
                self.table.upsert(entry, on_conflict="id").execute()
                success += 1
            except Exception as e2:
                logger.error(f"  Error importing {entry.get('id', 'unknown')}: {e2}")
                failed += 1
                self._log_failed_batch([entry], e2)

        return success, failed

    def _log_failed_batch(
        self, entries: list[dict[str, Any]], error: Exception | None
    ) -> None:
        """Log failed products to a local file for later inspection."""
        try:
            log_path = Path(FAILED_BATCHES_LOG)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).isoformat()
            with open(log_path, "a") as f:
                f.write(f"\n--- Failed batch at {timestamp} ---\n")
                f.write(f"Error: {error}\n")
                for entry in entries:
                    f.write(
                        json.dumps(
                            {
                                "id": entry.get("id"),
                                "title": entry.get("title"),
                                "product_url": entry.get("product_url"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            logger.info(f"  Logged {len(entries)} failed products to {log_path}")
        except Exception as log_err:
            logger.error(f"Failed to write error log: {log_err}")

    def _delete_products(
        self, products: list[dict[str, Any]]
    ) -> int:
        """Delete a list of products from the database."""
        if not products:
            return 0

        ids = [p["id"] for p in products if p.get("id")]
        if not ids:
            return 0

        try:
            self.table.delete().in_("id", ids).execute()
            logger.info(f"  Deleted {len(ids)} stale products")
            return len(ids)
        except Exception as e:
            logger.error(f"Error deleting stale products: {e}")
            # Fall back to individual deletes
            deleted = 0
            for pid in ids:
                try:
                    self.table.delete().eq("id", pid).execute()
                    deleted += 1
                except Exception as e2:
                    logger.error(f"Error deleting {pid}: {e2}")
            return deleted

    def run_import(
        self, products: list[dict[str, Any]]
    ) -> dict[str, int]:
        """
        Run the full import pipeline: classify, upsert, clean stale, and print summary.

        Args:
            products: List of product dicts (may or may not have embeddings populated)

        Returns:
            Dict with stats: new, updated, unchanged, stale_deleted, failed
        """
        # Step 1: Classify products
        classification = self.classify_products(products)

        # Step 2: Prepare entries for upsert
        # New + changed products need upserting
        to_upsert = []
        to_upsert.extend(classification["new"])
        to_upsert.extend(classification["changed"])

        # Prepare all entries
        entries = [self._prepare_entry(p) for p in to_upsert]

        # Step 3: Batch upsert
        total_success = 0
        total_failed = 0
        total_entries = len(entries)

        if total_entries > 0:
            total_batches = (total_entries + BATCH_SIZE - 1) // BATCH_SIZE
            logger.info(
                f"Upserting {total_entries} products in {total_batches} batch(es) "
                f"of up to {BATCH_SIZE}..."
            )

            for i in range(0, total_entries, BATCH_SIZE):
                batch = entries[i : i + BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                logger.info(
                    f"  Batch {batch_num}/{total_batches}: "
                    f"products {i + 1}-{min(i + BATCH_SIZE, total_entries)}"
                )

                success, failed = self._batch_upsert_with_retry(
                    batch, batch_num, total_batches
                )
                total_success += success
                total_failed += failed

                # Small delay between batches
                if i + BATCH_SIZE < total_entries:
                    time.sleep(EMBEDDING_DELAY)
        else:
            logger.info("No products need upserting.")

        # Step 4: Mark stale products that missed their first run
        marked = 0
        if classification["stale_to_mark"]:
            logger.info(f"Marking {len(classification['stale_to_mark'])} products as stale...")
            for i in range(0, len(classification["stale_to_mark"]), BATCH_SIZE):
                batch = classification["stale_to_mark"][i : i + BATCH_SIZE]
                for entry in batch:
                    try:
                        self.table.upsert(entry, on_conflict="id").execute()
                        marked += 1
                    except Exception as e:
                        logger.error(f"Error marking stale {entry.get('id')}: {e}")

        # Step 5: Delete stale products (missed 2+ runs)
        deleted = self._delete_products(classification["stale_to_delete"])

        # Step 6: Build and print summary
        stats = {
            "new": len(classification["new"]),
            "updated": len(classification["changed"]),
            "unchanged": len(classification["unchanged"]),
            "stale_deleted": deleted,
            "stale_marked": marked,
            "failed": total_failed,
        }

        self._print_summary(stats)
        return stats

    @staticmethod
    def _print_summary(stats: dict[str, int]) -> None:
        """Print a formatted run summary."""
        total_changed = stats["new"] + stats["updated"]
        total_processed = total_changed + stats["unchanged"]

        logger.info("=" * 60)
        logger.info("RUN SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  {stats['new']:>5} new products added")
        logger.info(f"  {stats['updated']:>5} products updated")
        logger.info(f"  {stats['unchanged']:>5} products unchanged (skipped)")
        logger.info(f"  {stats['stale_deleted']:>5} stale products deleted")
        logger.info(f"  {stats['stale_marked']:>5} products marked as stale (first missed run)")
        if stats["failed"] > 0:
            logger.info(f"  {stats['failed']:>5} products FAILED to import")
        logger.info("  " + "-" * 30)
        logger.info(f"  {total_changed:>5} total mutated ({total_processed} processed)")
        logger.info("=" * 60)

        if stats["failed"] > 0:
            logger.warning(
                f"Check {FAILED_BATCHES_LOG} for details on failed products."
            )


# Convenience function for the orchestrator
def run_smart_import(products: list[dict[str, Any]]) -> dict[str, int]:
    """
    Convenience wrapper: initialize SupabaseImporter and run the full import pipeline.
    Returns stats dict with keys: new, updated, unchanged, stale_deleted, failed.
    """
    importer = SupabaseImporter()
    return importer.run_import(products)
