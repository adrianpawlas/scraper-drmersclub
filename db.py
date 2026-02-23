"""
Supabase PostgREST HTTP client for smart sync import.
Uses plain HTTP (no Supabase JS client) for reliability in CI and batching.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class SupabaseREST:
    """PostgREST HTTP client for products table with smart sync logic."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.rest_url = f"{self.base_url}/rest/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        })

    def _normalize_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure every product has the same keys (PostgREST requirement). Use None for missing."""
        if not products:
            return []
        all_keys: set = set()
        for p in products:
            all_keys.update(p.keys())
        return [{k: p.get(k) for k in all_keys} for p in products]

    def upsert_new_only(
        self,
        products: List[Dict[str, Any]],
        chunk_size: int = 100,
    ) -> tuple[int, int]:
        """
        Insert new products only. Existing products (by source, product_url) are left untouched.
        Uses resolution=ignore-duplicates so we never overwrite existing rows.
        Returns (inserted_count, failed_count).
        """
        if not products:
            return 0, 0

        normalized = self._normalize_products(products)
        endpoint = f"{self.rest_url}/products"
        prefer = "resolution=ignore-duplicates,return=minimal"

        inserted = 0
        failed = 0

        for i in range(0, len(normalized), chunk_size):
            chunk = normalized[i : i + chunk_size]
            try:
                resp = self.session.post(
                    endpoint,
                    headers={**self.session.headers, "Prefer": prefer},
                    data=json.dumps(chunk),
                    timeout=90,
                )
                if resp.status_code in (200, 201, 204):
                    inserted += len(chunk)
                    logger.debug(f"Inserted chunk {i // chunk_size + 1}: {len(chunk)} products")
                else:
                    failed += len(chunk)
                    logger.error(f"Batch failed {resp.status_code}: {resp.text[:500]}")
                    # Retry one-by-one to avoid one bad row blocking the rest
                    for single in chunk:
                        r = self.session.post(
                            endpoint,
                            headers={**self.session.headers, "Prefer": prefer},
                            data=json.dumps([single]),
                            timeout=60,
                        )
                        if r.status_code in (200, 201, 204):
                            inserted += 1
                            failed -= 1
                        else:
                            logger.error(f"Single insert failed: {r.status_code} {r.text[:200]}")
            except requests.RequestException as e:
                failed += len(chunk)
                logger.error(f"Request failed: {e}", exc_info=True)

        return inserted, failed

    def delete_stale_products(self, source: str, keep_product_ids: List[str]) -> tuple[int, int]:
        """
        Delete products from this source that are NOT in keep_product_ids.
        Use deterministic product ids (e.g. drmersclub_123) for keep_product_ids.
        Returns (deleted_count, error_count).
        """
        if not keep_product_ids:
            logger.warning("No keep_product_ids provided; skipping delete to avoid wiping all products")
            return 0, 0

        endpoint = f"{self.rest_url}/products"
        batch_size = 500
        total_deleted = 0
        errors = 0

        # Fetch existing ids for this source, compute stale = existing - keep, delete stale in batches
        try:
            # Fetch all product ids for this source (paginated - PostgREST defaults to 1000)
            existing_rows: List[Dict[str, Any]] = []
            offset = 0
            limit = 1000
            while True:
                get_resp = self.session.get(
                    endpoint,
                    params={"source": f"eq.{source}", "select": "id"},
                    headers={**self.session.headers, "Range-Unit": "items", "Range": f"{offset}-{offset + limit - 1}"},
                    timeout=60,
                )
                if get_resp.status_code != 200:
                    logger.error(f"Failed to fetch existing ids: {get_resp.status_code}")
                    return 0, 1
                page = get_resp.json()
                existing_rows.extend(page)
                if len(page) < limit:
                    break
                offset += limit
            existing_ids = {r["id"] for r in existing_rows}
            keep_set = set(keep_product_ids)
            stale_ids = list(existing_ids - keep_set)

            if not stale_ids:
                logger.info("No stale products to delete")
                return 0, 0

            logger.info(f"Deleting {len(stale_ids)} stale products (not in current catalog)")

            for i in range(0, len(stale_ids), batch_size):
                batch = stale_ids[i : i + batch_size]
                in_list = ",".join(batch)
                try:
                    # Delete where source=eq.X and id=in.(stale1,stale2,...)
                    resp = self.session.delete(
                        endpoint,
                        params={
                            "source": f"eq.{source}",
                            "id": f"in.({in_list})",
                        },
                        timeout=90,
                    )
                    if resp.status_code in (200, 204):
                        total_deleted += len(batch)
                    else:
                        errors += 1
                        logger.error(f"Delete batch failed {resp.status_code}: {resp.text[:300]}")
                except requests.RequestException as e:
                    errors += 1
                    logger.error(f"Delete failed: {e}")

            return total_deleted, errors

        except requests.RequestException as e:
            logger.error(f"Failed to fetch existing products: {e}", exc_info=True)
            return 0, 1
