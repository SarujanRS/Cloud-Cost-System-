"""Refresh the cached pricing catalogue from the live provider APIs.

Usage:  python pricing/refresh.py
Azure refreshes with no setup. AWS uses boto3 if credentials are configured.
GCP uses the Cloud Billing Catalog API if GCP_API_KEY is set. Any provider that
cannot be reached keeps its cached rates.
"""
import json, datetime as dt
from pathlib import Path
from providers import fetch_all  # when run from inside pricing/
CACHE = Path(__file__).with_name("catalog_cache.json")

def main():
    cache = json.load(open(CACHE))
    live = fetch_all()
    for prov in ["AWS", "Azure", "GCP"]:
        res = live[prov]; region = res["region"]
        node = cache["providers"][prov]["regions"].get(region)
        if node is not None and res["source"] == "live":
            node["instances"] = res["instances"]
        cache["providers"][prov]["source"] = res["source"]
        print(f"{prov:6s} {res['source']:5s} region={region} "
              f"latency={res.get('latency_ms')}ms "
              f"{'' if not res.get('error') else '(' + res['error'][:60] + ')'}")
    cache["updated"] = dt.datetime.utcnow().isoformat() + "Z"
    json.dump(cache, open(CACHE, "w"), indent=2)
    print("Cache updated:", cache["updated"])

if __name__ == "__main__":
    main()
