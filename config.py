"""Configuration for DRMERS CLUB scraper."""

import os
from dotenv import load_dotenv

load_dotenv()

# Scraper settings
BASE_URL = "https://drmersclub.com"
COLLECTION_URL = f"{BASE_URL}/collections/shop-all"
PRODUCTS_JSON_URL = f"{COLLECTION_URL}/products.json"
PRODUCTS_PER_PAGE = 250

# Brand & source
SOURCE = "scraper"
BRAND = "Drmers Club"

# Embedding model (siglip-base outputs 768-dim; "384" = image resolution)
SIGLIP_MODEL = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768  # Must match Supabase vector column dimension

# Supabase (set via env or .env; for GitHub Actions use repository secrets)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://yqawmzggcgpeyaaynrjk.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_KEY"))

# Currency conversion (optional - store uses CAD, add approximate conversions)
# Format: {target_currency: (rate_from_cad, currency_code)}
CURRENCY_CONVERSIONS = {
    "CAD": (1.0, "CAD"),
    "USD": (0.72, "USD"),
    "EUR": (0.67, "EUR"),
    "GBP": (0.57, "GBP"),
    "CZK": (16.5, "CZK"),
    "PLN": (2.9, "PLN"),
}
