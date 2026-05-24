"""
Scraper module for Drmers Club Shopify store.
Fetches all products via Shopify's public JSON API.
"""

import json
import logging
import re
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

from config import (
    BRAND,
    COLLECTION_URL,
    EXCHANGE_API_URL,
    MAX_RETRIES,
    PRIMARY_CURRENCY,
    PRODUCTS_LISTING_URL,
    REQUEST_TIMEOUT,
    RETRY_DELAY,
    SOURCE,
    STORE_URL,
    TARGET_CURRENCIES,
)

logger = logging.getLogger(__name__)


def _request_with_retry(url: str, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
    """Make a GET request with retry logic."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_error = e
            logger.warning(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {url} - {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    raise last_error  # type: ignore


def fetch_all_products() -> list[dict[str, Any]]:
    """
    Fetch all products from the collection using Shopify's products.json API.
    Shopify supports up to 250 products per page.
    """
    all_products: list[dict[str, Any]] = []
    page = 1

    while True:
        url = f"{PRODUCTS_LISTING_URL}?page={page}&limit=250"
        logger.info(f"Fetching products page {page}...")
        resp = _request_with_retry(url)
        data = resp.json()

        products = data.get("products", [])
        if not products:
            break

        all_products.extend(products)
        logger.info(f"  Got {len(products)} products on page {page}")
        page += 1

        # Safety check - if we got fewer than 250, it's the last page
        if len(products) < 250:
            break

    logger.info(f"Total products fetched: {len(all_products)}")
    return all_products


def fetch_product_details(handle: str) -> dict[str, Any]:
    """Fetch detailed info for a single product by its handle."""
    url = f"{STORE_URL}/products/{handle}.json"
    resp = _request_with_retry(url)
    data = resp.json()
    return data.get("product", {})


def _clean_html(html: str | None) -> str:
    """Strip HTML tags and clean up whitespace from a description."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_exchange_rates() -> dict[str, float]:
    """Fetch current exchange rates from CAD to target currencies."""
    try:
        resp = _request_with_retry(EXCHANGE_API_URL)
        data = resp.json()
        rates = data.get("rates", {})
        return {cur: rates[cur] for cur in TARGET_CURRENCIES if cur in rates}
    except Exception as e:
        logger.warning(f"Failed to fetch exchange rates: {e}")
        return {}


def _format_price_string(cad_price: float, rates: dict[str, float]) -> str:
    """
    Convert CAD price to multiple currencies and format as "XX.XXEUR,YY.YYUSD,..."
    Prices are rounded to 2 decimal places.
    Falls back to CAD if exchange rates are unavailable.
    """
    parts = []
    for currency in TARGET_CURRENCIES:
        rate = rates.get(currency)
        if rate:
            converted = round(cad_price * rate, 2)
            parts.append(f"{converted:.2f}{currency}")

    if parts:
        return ",".join(parts)
    # Fallback: show CAD price if no exchange rates available
    return f"{cad_price:.2f}{PRIMARY_CURRENCY}"


def _extract_category(product: dict[str, Any]) -> str | None:
    """
    Extract category from product_type and tags.
    Returns a comma-separated string of categories.
    """
    categories = []

    product_type = product.get("product_type", "").strip().lower()
    if product_type:
        # Map product_type to a cleaner category name
        type_mapping = {
            "zip up": "Zip-Up Hoodies",
            "outerwear": "Outerwear",
            "bottom": "Bottoms",
            "top": "Tops",
            "tee": "T-Shirts",
            "knit": "Knitwear",
            "sweater": "Sweaters",
            "hoodie": "Hoodies",
            "hat": "Hats",
            "headwear": "Hats",
            "pant": "Pants",
            "pants": "Pants",
            "short": "Shorts",
            "shorts": "Shorts",
            "accessory": "Accessories",
            "bag": "Bags",
        }
        mapped = type_mapping.get(product_type)
        if mapped:
            categories.append(mapped)
        else:
            categories.append(product_type.capitalize())

    # Also derive from tags
    tags = [t.lower() for t in product.get("tags", [])]
    tag_category_map = {
        "hoodie": "Hoodies",
        "sweater": "Sweaters",
        "tee": "T-Shirts",
        "knit": "Knitwear",
        "denim": "Denim",
        "jacket": "Jackets",
        "outerwear": "Outerwear",
        "pant": "Pants",
        "short": "Shorts",
        "hat": "Hats",
        "accessory": "Accessories",
    }

    for tag in tags:
        for keyword, cat in tag_category_map.items():
            if keyword in tag and cat not in categories:
                categories.append(cat)
                break

    return ", ".join(categories) if categories else None


def _extract_gender(product: dict[str, Any]) -> str | None:
    """Extract gender from product tags or title."""
    tags = [t.lower() for t in product.get("tags", [])]
    title = product.get("title", "").lower()

    if "women" in tags or "women" in title or "womens" in title:
        return "women"
    elif "men" in tags or "men" in title or "mens" in title or "unisex" in tags:
        return "men"
    else:
        return "unisex"


def _get_sizes(product: dict[str, Any]) -> str | None:
    """Extract available sizes from variants."""
    sizes = []
    for variant in product.get("variants", []):
        title = variant.get("title", "")
        if title and title.lower() not in ("default title", "default"):
            sizes.append(title)

    # Deduplicate while preserving order
    seen = set()
    unique_sizes = []
    for s in sizes:
        if s not in seen:
            seen.add(s)
            unique_sizes.append(s)

    return ", ".join(unique_sizes) if unique_sizes else None


def process_product(
    product: dict[str, Any],
    rates: dict[str, float],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Process a raw Shopify product into our database schema.

    Args:
        product: Product dict from the listing API
        rates: Exchange rates from CAD
        details: Optional detailed product data from the individual endpoint

    Returns:
        A dict ready for Supabase insertion
    """
    # Merge listing data with details if available
    merged = product.copy()
    if details:
        merged.update(details)

    shopify_id = str(merged["id"])
    handle = merged["handle"]

    # Images
    images = merged.get("images", [])
    image_url = images[0]["src"] if images else None
    additional_images_list = [img["src"] for img in images[1:]] if len(images) > 1 else []
    additional_images = " , ".join(additional_images_list) if additional_images_list else None

    # Variants - get first variant's price info
    variants = merged.get("variants", [])
    first_variant = variants[0] if variants else {}

    # Pricing logic
    # In Shopify:
    # - price = current selling price
    # - compare_at_price = original/regular price (shown as strikethrough when on sale)
    cad_price = float(first_variant.get("price", 0))
    cad_compare_at = first_variant.get("compare_at_price")

    # Determine original price vs sale price
    if cad_compare_at and float(cad_compare_at) > cad_price:
        # On sale: compare_at_price is original, price is sale
        original_cad = float(cad_compare_at)
        sale_cad = cad_price
    else:
        # Not on sale or compare_at_price equals price
        original_cad = cad_price
        sale_cad = None

    # Convert prices to multi-currency format
    price_str = _format_price_string(original_cad, rates) if original_cad > 0 else None
    sale_str = _format_price_string(sale_cad, rates) if sale_cad else None

    # Clean description
    description = _clean_html(merged.get("body_html", ""))

    # Category
    category = _extract_category(merged)

    # Gender
    gender = _extract_gender(merged)

    # Sizes
    sizes = _get_sizes(merged)

    # Tags
    tags = merged.get("tags", [])

    # Build metadata
    metadata = {
        "shopify_id": shopify_id,
        "handle": handle,
        "vendor": merged.get("vendor", ""),
        "product_type": merged.get("product_type", ""),
        "options": merged.get("options", []),
        "variants": [
            {
                "id": v["id"],
                "title": v["title"],
                "sku": v.get("sku", ""),
                "price": v["price"],
                "compare_at_price": v.get("compare_at_price"),
                "available": v.get("available", True),
                "grams": v.get("grams", 0),
            }
            for v in variants
        ],
        "currency": PRIMARY_CURRENCY,
        "prices_cad": {
            "original": original_cad,
            "sale": sale_cad,
        },
        "prices_all": {
            "original": price_str,
            "sale": sale_str,
        },
        "num_images": len(images),
    }

    # Build text for info_embedding
    info_text_parts = [
        f"Title: {merged['title']}",
        f"Brand: {BRAND}",
        f"Category: {category or ''}",
        f"Gender: {gender or ''}",
        f"Price: {price_str or ''}",
        f"Sale: {sale_str or ''}" if sale_str else "",
        f"Description: {description}" if description else "",
        f"Sizes: {sizes or ''}" if sizes else "",
        f"Tags: {', '.join(tags)}" if tags else "",
    ]
    info_text = ". ".join(p for p in info_text_parts if p)

    result: dict[str, Any] = {
        "id": f"drmersclub_{shopify_id}",
        "source": SOURCE,
        "product_url": f"https://drmersclub.com/products/{handle}",
        "affiliate_url": None,
        "image_url": image_url,
        "brand": BRAND,
        "title": merged["title"],
        "description": description if description else None,
        "category": category,
        "gender": gender,
        "size": sizes,
        "second_hand": False,
        "price": price_str,
        "sale": sale_str,
        "tags": tags if tags else None,
        "additional_images": additional_images,
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "image_embedding": None,  # To be filled by embeddings module
        "info_embedding": None,  # To be filled by embeddings module
        "info_text": info_text,  # Temporary field for embedding generation
        "country": "CA",
        "compressed_image_url": None,
        "other": None,
    }

    return result


def fetch_and_process_all_products() -> list[dict[str, Any]]:
    """Fetch all products from the store and process them into our schema."""
    logger.info("Fetching exchange rates...")
    rates = _get_exchange_rates()
    logger.info(f"Exchange rates: {rates}")

    logger.info("Fetching all products from collection...")
    raw_products = fetch_all_products()
    logger.info(f"Fetched {len(raw_products)} raw products")

    logger.info("Processing products...")
    processed = []
    for i, product in enumerate(raw_products):
        try:
            processed_product = process_product(product, rates)
            processed.append(processed_product)
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(raw_products)} products")
        except Exception as e:
            logger.error(f"Error processing product {product.get('title')}: {e}")

    logger.info(f"Processed {len(processed)} products total")
    return processed



