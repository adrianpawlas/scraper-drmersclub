# DRMERS CLUB Scraper

Full scraper for [DRMERS CLUB](https://drmersclub.com) fashion store. Fetches all products, generates 768-dim embeddings via SigLIP, and imports to Supabase.

## Features

- **Shopify JSON API** – Uses `products.json` (no Playwright/headless browser needed)
- **SigLIP embeddings** – Image embeddings + text info embeddings (google/siglip-base-patch16-384)
- **Supabase import** – Upserts into `products` table with conflict resolution on `(source, product_url)`

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

Create `.env` with your Supabase credentials (see `.env.example`).

## Usage

```bash
# Full run (all products, embeddings, Supabase import)
python main.py

# Limit to 5 products (testing)
python main.py --limit 5

# Dry run (no DB writes)
python main.py --dry-run

# Skip embeddings (faster, for testing data flow)
python main.py --skip-embeddings
```

## Output Fields

| Field | Source |
|-------|--------|
| source | `"scraper"` |
| brand | `"Drmers Club"` |
| image_url | First product image |
| additional_images | `"url1 , url2 , url3"` |
| image_embedding | 768-dim SigLIP image embedding |
| info_embedding | 768-dim SigLIP text embedding (title, desc, category, etc.) |
| price | `"140.00CAD, 100.80USD, 93.80EUR, ..."` |
| gender | `"man"` |
| second_hand | `false` |
| category | From product_type + tags (e.g. "Hoodies, Tops") |

## GitHub Actions (daily + manual)

The scraper runs automatically every day at **midnight UTC** and can be triggered manually:

1. Go to **Settings → Secrets and variables → Actions**
2. Add:
   - `SUPABASE_URL` = your Supabase project URL
   - `SUPABASE_SERVICE_KEY` = your Supabase service_role key

3. **Manual run:** Actions → Run DRMERS CLUB Scraper → Run workflow

To change the schedule (e.g. midnight in your timezone), edit `.github/workflows/run-scraper.yml` and the cron expression.

## Requirements

- Python 3.10+
- ~2GB RAM for SigLIP model
- GPU optional (faster embeddings)
