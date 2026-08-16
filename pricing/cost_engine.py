"""
cost_engine.py
--------------
Turns a live pricing catalogue plus a workload definition into a per-provider
cost breakdown and a cross-provider comparison. The cost is computed directly
from the (live or cached) rates, so every figure is transparent and traceable
to a published price - this is the rule-based "live cost". The machine-learning
model provides a separate predicted figure for comparison.
"""

import json
import re
import datetime as dt
from pathlib import Path

CACHE_PATH = Path(__file__).with_name("catalog_cache.json")
DEFAULT_REGIONS = {"AWS": "eu-west-2", "Azure": "uksouth", "GCP": "europe-west2"}
REGION_ALIASES = {
    "AWS": {
        "us-east-1": "us-east-1", "us-east1": "us-east-1",
        "eu-west-1": "eu-west-1", "eu-west1": "eu-west-1",
        "eu-west-2": "eu-west-2", "eu-west2": "eu-west-2",
        "us-east-2": "us-east-2", "us-east2": "us-east-2",
        "us-west-1": "us-west-1", "us-west2": "us-west-2",
        "us-west-2": "us-west-2", "ap-southeast-1": "ap-southeast-1",
    },
    "Azure": {
        "eastus": "eastus", "eastus2": "eastus2",
        "westeurope": "westeurope", "uksouth": "uksouth",
        "centralus": "centralus", "westus": "westus",
        "westus2": "westus2", "westus3": "westus3",
    },
    "GCP": {
        "europe-west1": "europe-west1", "europe-west-1": "europe-west1",
        "europe-west2": "europe-west2", "europe-west-2": "europe-west2",
        "us-central1": "us-central1", "us-central-1": "us-central1",
        "us-east1": "us-east1", "us-east-1": "us-east1",
        "asia-southeast1": "asia-southeast1",
    },
}


def _default_catalog():
    """Return a small but valid seeded catalogue used when the cache file is missing."""
    return {
        "currency": "USD",
        "updated": dt.datetime.utcnow().isoformat() + "Z",
        "providers": {
            "AWS": {
                "source": "cache",
                "api": "AWS Price List Bulk API",
                "regions": {
                    "eu-west-2": {
                        "label": "London",
                        "instances": [
                            {"instance_type": "t3.medium", "vcpu": 2, "ram_gb": 4, "price_hr": 0.0472},
                            {"instance_type": "t3.large", "vcpu": 2, "ram_gb": 8, "price_hr": 0.0944},
                            {"instance_type": "m6i.large", "vcpu": 2, "ram_gb": 8, "price_hr": 0.111},
                            {"instance_type": "m6i.xlarge", "vcpu": 4, "ram_gb": 16, "price_hr": 0.222},
                            {"instance_type": "m6i.2xlarge", "vcpu": 8, "ram_gb": 32, "price_hr": 0.444},
                            {"instance_type": "m6i.4xlarge", "vcpu": 16, "ram_gb": 64, "price_hr": 0.888},
                        ],
                        "storage_gb_month": 0.0928,
                        "egress_gb": 0.09,
                        "managed_db_month": 62.0,
                        "load_balancer_month": 18.0,
                    }
                },
            },
            "Azure": {
                "source": "cache",
                "api": "Azure Retail Prices API",
                "regions": {
                    "uksouth": {
                        "label": "UK South",
                        "instances": [
                            {"instance_type": "B2ms", "vcpu": 2, "ram_gb": 8, "price_hr": 0.0928},
                            {"instance_type": "D2s_v5", "vcpu": 2, "ram_gb": 8, "price_hr": 0.107},
                            {"instance_type": "D4s_v5", "vcpu": 4, "ram_gb": 16, "price_hr": 0.214},
                            {"instance_type": "D8s_v5", "vcpu": 8, "ram_gb": 32, "price_hr": 0.428},
                            {"instance_type": "D16s_v5", "vcpu": 16, "ram_gb": 64, "price_hr": 0.856},
                        ],
                        "storage_gb_month": 0.0805,
                        "egress_gb": 0.087,
                        "managed_db_month": 59.0,
                        "load_balancer_month": 16.0,
                    }
                },
            },
            "GCP": {
                "source": "cache",
                "api": "GCP Cloud Billing Catalog API",
                "regions": {
                    "europe-west2": {
                        "label": "London",
                        "instances": [
                            {"instance_type": "e2-medium", "vcpu": 2, "ram_gb": 4, "price_hr": 0.0367},
                            {"instance_type": "e2-standard-2", "vcpu": 2, "ram_gb": 8, "price_hr": 0.0734},
                            {"instance_type": "n2-standard-2", "vcpu": 2, "ram_gb": 8, "price_hr": 0.1077},
                            {"instance_type": "n2-standard-4", "vcpu": 4, "ram_gb": 16, "price_hr": 0.2154},
                            {"instance_type": "n2-standard-8", "vcpu": 8, "ram_gb": 32, "price_hr": 0.4308},
                            {"instance_type": "n2-standard-16", "vcpu": 16, "ram_gb": 64, "price_hr": 0.8616},
                        ],
                        "storage_gb_month": 0.085,
                        "egress_gb": 0.085,
                        "managed_db_month": 63.0,
                        "load_balancer_month": 18.5,
                    }
                },
            },
        },
    }


def _normalize_region_key(region):
    if region is None:
        return ""
    value = str(region).strip().lower().replace("_", "-").replace(" ", "")
    if not value:
        return ""
    return re.sub(r"-(\d)$", r"\1", value)


def resolve_provider_region(catalog, provider, region):
    """Map a UI or API region value to a valid region key in the provider catalogue."""
    provider_regions = catalog["providers"].get(provider, {}).get("regions", {})
    if not provider_regions:
        return region
    if region in provider_regions:
        return region

    lookup = _normalize_region_key(region)
    if lookup in provider_regions:
        return lookup

    exact_alias = REGION_ALIASES.get(provider, {}).get(str(region).strip())
    if exact_alias and exact_alias in provider_regions:
        return exact_alias

    alias_map = REGION_ALIASES.get(provider, {})
    for alias_key, canonical in alias_map.items():
        if _normalize_region_key(alias_key) == lookup and canonical in provider_regions:
            return canonical

    for key in provider_regions:
        if _normalize_region_key(key) == lookup:
            return key

    return DEFAULT_REGIONS.get(provider, next(iter(provider_regions)))


def _region_rates(catalog, provider, region):
    resolved = resolve_provider_region(catalog, provider, region)
    return catalog["providers"][provider]["regions"][resolved]


def egress_cost(rate, gb):
    """First 100 GB free, then per-GB rate, with a volume discount above 10 TB."""
    billable = max(0.0, gb - 100.0)
    tier1 = min(billable, 10_000.0) * rate
    tier2 = max(0.0, billable - 10_000.0) * rate * 0.85
    return tier1 + tier2


def cost_for_instance(inst, rates, workload):
    """Full monthly cost breakdown for one instance type under a workload."""
    compute = inst["price_hr"] * workload["monthly_hours"] * workload["instance_count"]
    storage = rates["storage_gb_month"] * workload["storage_gb"]
    egress = egress_cost(rates["egress_gb"], workload["egress_gb"])
    db = rates["managed_db_month"] if workload.get("managed_db") else 0.0
    lb = rates["load_balancer_month"] if workload.get("load_balancer") else 0.0
    total = compute + storage + egress + db + lb
    return {
        "instance_type": inst["instance_type"], "vcpu": inst["vcpu"],
        "ram_gb": inst["ram_gb"], "price_hr": inst["price_hr"],
        "breakdown": {"compute": round(compute, 2), "storage": round(storage, 2),
                      "egress": round(egress, 2), "managed_db": round(db, 2),
                      "load_balancer": round(lb, 2)},
        "monthly_cost": round(total, 2),
    }


def best_for_provider(catalog, provider, region, workload):
    """Cheapest instance in a provider/region that meets the requirements."""
    resolved_region = resolve_provider_region(catalog, provider, region)
    rates = _region_rates(catalog, provider, resolved_region)
    viable = [i for i in rates["instances"]
              if i["vcpu"] >= workload["min_vcpu"] and i["ram_gb"] >= workload["min_ram_gb"]]
    if not viable:
        return None
    options = [cost_for_instance(i, rates, workload) for i in viable]
    options.sort(key=lambda o: o["monthly_cost"])
    best = dict(options[0])  # copy so best is not contained in its own all_options
    best["provider"] = provider
    best["region"] = resolved_region
    best["region_label"] = rates.get("label", resolved_region)
    best["all_options"] = options
    best["source"] = catalog["providers"][provider].get("source", "cache")
    return best


def compare_providers(catalog, workload, regions=None):
    """Compare the best option from each provider and rank them."""
    regions = regions or DEFAULT_REGIONS.copy()
    resolved_regions = {prov: resolve_provider_region(catalog, prov, regions.get(prov, DEFAULT_REGIONS.get(prov))) for prov in ["AWS", "Azure", "GCP"]}
    results = []
    for prov in ["AWS", "Azure", "GCP"]:
        best = best_for_provider(catalog, prov, resolved_regions[prov], workload)
        if best:
            results.append(best)
    results.sort(key=lambda r: r["monthly_cost"])
    if not results:
        return {"error": "no viable option for the requirements"}
    cheapest, dearest = results[0], results[-1]
    saving_pct = round((dearest["monthly_cost"] - cheapest["monthly_cost"])
                       / dearest["monthly_cost"] * 100, 1) if dearest["monthly_cost"] else 0
    return {
        "recommended": {"provider": cheapest["provider"],
                        "instance_type": cheapest["instance_type"],
                        "region": cheapest["region_label"],
                        "monthly_cost": cheapest["monthly_cost"]},
        "saving_vs_dearest_pct": saving_pct,
        "providers": results,
    }


def load_catalog(path=CACHE_PATH):
    """Load the cached catalogue, creating a valid seed if the cache file is missing."""
    if not path.exists():
        catalog = _default_catalog()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2)
        except OSError:
            pass
        return catalog

    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    cat = load_catalog()
    wl = {"min_vcpu": 4, "min_ram_gb": 16, "instance_count": 4, "monthly_hours": 730,
          "storage_gb": 500, "egress_gb": 1500, "managed_db": 1, "load_balancer": 1}
    import pprint
    result = compare_providers(cat, wl)
    print("Recommended:", result["recommended"])
    print("Saving vs dearest: {}%\n".format(result["saving_vs_dearest_pct"]))
    for p in result["providers"]:
        print(f"{p['provider']:6s} {p['instance_type']:14s} ${p['monthly_cost']:>9.2f}  "
              f"({p['region_label']}, {p['source']})  {p['breakdown']}")
