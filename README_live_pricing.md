# Multi-Cloud Cost Console — live pricing system

A console that compares AWS, Azure and Google Cloud on **live on-demand pricing**
for a defined workload, and cross-checks the result against the machine-learning
cost model from the dissertation.

## What it does
- Fetches live instance pricing from each provider's public pricing API.
- Computes the full monthly cost (compute + storage + egress + managed DB + load
  balancer) for the cheapest instance type that meets your requirements, per
  provider, and ranks them.
- Shows the ML-predicted cost alongside the live rule-based cost.

## Pricing sources
| Provider | API | Setup |
|----------|-----|-------|
| Azure | Retail Prices API (`prices.azure.com`) | none — public, no auth |
| AWS | Price List Query API (via boto3) | AWS credentials (`aws configure`) |
| GCP | Cloud Billing Catalog API | free API key in `GCP_API_KEY` |

Any provider that can't be reached falls back to the cached rate catalogue in
`pricing/catalog_cache.json`, so the console always has data.

## Run it
```bash
pip install -r requirements.txt
python backend/app.py            # serves the console at http://localhost:5000
```
Open http://localhost:5000, set a workload, and select **Compare live costs**.
Use **Refresh live prices** to pull current rates.

To refresh the cached catalogue from the command line:
```bash
cd pricing && python refresh.py
```

### Optional: enable live AWS and GCP
```bash
aws configure                    # AWS credentials for the Pricing Query API
export GCP_API_KEY=your_key      # GCP Cloud Billing Catalog API key
```

## Standalone preview
`multicloud_cost_console.html` opens in any browser without the backend, using
the embedded rate catalogue (marked CACHED). Run the backend for live prices and
ML forecasts.

## Files
```
backend/app.py                 Flask API (providers, compare, predict, refresh)
backend/static/index.html      the console dashboard
pricing/providers.py           live fetchers (AWS / Azure / GCP) + fallback
pricing/cost_engine.py         cost model + cross-provider comparison
pricing/catalog_cache.json     cached on-demand rates (seed + fallback)
pricing/refresh.py             refresh the cache from live APIs
ml/model.joblib                trained Gradient Boosting model (ML forecast)
```
