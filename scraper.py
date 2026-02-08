"""DRMERS CLUB product scraper - fetches products from Shopify JSON API."""

import json
import logging
import re
import time
from html import unescape
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import (
    BASE_URL,
    BRAND,
    CURRENCY_CONVERSIONS,
    PRODUCTS_PER_PAGE,
    PRODUCTS_JSON_URL,
    SOURCE,
)

logger = logging.getLogger(__name__)


def strip_html(html: str) -> str:
    """Strip HTML tags and decode entities."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return unescape(soup.get_text(separator=" ", strip=True))


def format_category(product_type: str, tags: List[str]) -> str:
    """Format category from product_type and tags. Multi-category as comma-separated."""
    categories = set()
    if product_type:
        # Normalize: "zip up hoodie" -> "Zip Up, Hoodie"
        parts = [p.strip().title() for p in re.split(r"[\s&]+", product_type) if p]
        categories.update(parts)

    # Map common tags to categories
    tag_to_cat = {
        "hoodie": "Hoodies",
        "hoodies": "Hoodies",
        "denim": "Denim",
        "jeans": "Denim",
        "knitwear": "Knitwear",
        "sweater": "Knitwear",
        "tee": "Tees",
        "tees": "Tees",
        "longsleeve": "Longsleeves",
        "sweatpants": "Sweatpants",
        "jacket": "Jackets",
        "zip up": "Hoodies",
        "basics": "Basics",
        "tops": "Tops",
        "bottoms": "Bottoms",
        "small goods": "Small Goods",
    }
    for tag in (t.lower() for t in tags or []):
        if tag in tag_to_cat:
            categories.add(tag_to_cat[tag])

    return ", ".join(sorted(categories)) if categories else (product_type or "")


def format_price(variants: List[Dict]) -> str:
    """Format price with CAD + converted currencies. Store uses CAD."""
    if not variants:
        return ""
    prices = set()
    for v in variants:
        try:
            p = float(v.get("price", 0))
            if p <= 0:
                continue
            prices.add(p)
        except (TypeError, ValueError):
            pass

    if not prices:
        return ""

    cad_price = min(prices)  # Use lowest variant price
    parts = [f"{cad_price:.2f}CAD"]
    for key, (rate, code) in CURRENCY_CONVERSIONS.items():
        if key != "CAD":
            converted = round(cad_price * rate, 2)
            parts.append(f"{converted:.2f}{code}")
    return ",".join(parts)  # "140.00CAD,100.80USD,93.80EUR,..."


def format_sale(variants: List[Dict]) -> Optional[str]:
    """Return sale info if compare_at_price > price."""
    for v in variants:
        try:
            price = float(v.get("price", 0))
            compare = v.get("compare_at_price")
            if compare is not None:
                compare = float(compare)
                if compare > price:
                    return f"Sale: {price:.2f}CAD (was {compare:.2f}CAD)"
        except (TypeError, ValueError):
            pass
    return None


def format_sizes(variants: List[Dict]) -> str:
    """Extract size options from variants."""
    sizes = []
    for v in variants:
        opt = v.get("option1") or v.get("title")
        if opt and opt not in sizes:
            sizes.append(str(opt))
    return ", ".join(sizes) if sizes else ""


def fetch_all_products() -> List[Dict[str, Any]]:
    """Fetch all products from Shopify collection JSON API (paginated)."""
    all_products = []
    page = 1
    while True:
        url = f"{PRODUCTS_JSON_URL}?limit={PRODUCTS_PER_PAGE}&page={page}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            products = data.get("products", [])
            if not products:
                break
            all_products.extend(products)
            logger.info(f"Fetched page {page}: {len(products)} products (total: {len(all_products)})")
            if len(products) < PRODUCTS_PER_PAGE:
                break
            page += 1
            time.sleep(0.5)  # Be polite
        except Exception as e:
            logger.error(f"Failed to fetch page {page}: {e}")
            break
    return all_products


def transform_product(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Transform Shopify product to our schema (without embeddings)."""
    product_id = str(raw.get("id", ""))
    handle = raw.get("handle", "")
    product_url = f"{BASE_URL}/collections/shop-all/products/{handle}" if handle else ""

    images = raw.get("images") or []
    image_urls = [img.get("src") for img in images if img.get("src")]
    image_url = image_urls[0] if image_urls else ""
    additional_images = " , ".join(image_urls[1:]) if len(image_urls) > 1 else None

    variants = raw.get("variants") or []
    description = strip_html(raw.get("body_html", ""))
    product_type = raw.get("product_type", "")
    tags_list = raw.get("tags") or []

    metadata = {
        "product_id": product_id,
        "handle": handle,
        "vendor": raw.get("vendor"),
        "product_type": product_type,
        "tags": tags_list,
        "published_at": raw.get("published_at"),
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "variants_count": len(variants),
    }

    return {
        "id": f"drmersclub_{product_id}",
        "source": SOURCE,
        "product_url": product_url,
        "affiliate_url": None,
        "image_url": image_url,
        "brand": BRAND,
        "title": raw.get("title", ""),
        "description": description or None,
        "category": format_category(product_type, tags_list) or None,
        "gender": "man",
        "metadata": json.dumps(metadata, default=str) if metadata else None,
        "size": format_sizes(variants) or None,
        "second_hand": False,
        "country": "CA",
        "tags": tags_list if tags_list else None,
        "price": format_price(variants) or None,
        "sale": format_sale(variants),
        "additional_images": additional_images,
        "other": None,
    }
