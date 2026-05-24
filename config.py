"""
Configuration for Drmers Club product scraper.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Store info
SOURCE = "scraper-drmersclub"
BRAND = "Drmers Club"
STORE_URL = "https://drmersclub.com"
COLLECTION_HANDLE = "shop-all"
COLLECTION_URL = f"{STORE_URL}/collections/{COLLECTION_HANDLE}"

# Shopify API (no auth needed for public products JSON endpoint)
PRODUCTS_LISTING_URL = f"{COLLECTION_URL}/products.json"
PRODUCT_DETAILS_URL = f"{STORE_URL}/products/{{handle}}.json"

# Pagination - Shopify allows up to 250 per page
PRODUCTS_PER_PAGE = 250

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://yqawmzggcgpeyaaynrjk.supabase.co")
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4",
)
SUPABASE_TABLE = "products"

# Embedding model
EMBEDDING_MODEL_ID = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768

# Exchange rates provider
EXCHANGE_API_URL = "https://api.exchangerate-api.com/v4/latest/CAD"

# Primary currency from Shopify API
PRIMARY_CURRENCY = "CAD"

# Currencies to convert to (in order of priority)
TARGET_CURRENCIES = ["EUR", "USD", "GBP", "CZK", "PLN", "AUD", "NZD", "SGD"]

# Request settings
REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 2

# Database batch size
BATCH_SIZE = 25
