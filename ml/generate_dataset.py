"""
generate_dataset.py
-------------------
Builds a synthetic multi-cloud deployment cost dataset.

Each row is one deployment scenario: a choice of cloud provider, region,
instance type, plus the workload's storage, egress, load-balancing and
managed-database requirements and a monthly usage profile. The target is the
estimated total monthly cost in USD.

The cost model is parametric and seeded with published 2025 on-demand pricing
for common general-purpose instance families across AWS, Azure and GCP. It is
deliberately additive at its core (compute + storage + egress + LB + managed DB)
with multiplicative regional adjustments and tiered egress pricing, so that a
linear model captures the broad trend while tree-based ensembles capture the
interactions and tiers.

NOTE: This is a MODELLED dataset, not harvested billing data. That is stated as
a limitation throughout the dissertation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 8000

# ---------------------------------------------------------------------------
# Instance catalogue: (provider, family, instance_type, vcpu, ram_gb, $/hour)
# Prices approximate published 2025 on-demand rates for general-purpose types.
# ---------------------------------------------------------------------------
CATALOGUE = [
    # AWS  (m6i / t3 general purpose)
    ("AWS",   "t3.medium",     2,   4,  0.0416),
    ("AWS",   "t3.large",      2,   8,  0.0832),
    ("AWS",   "m6i.large",     2,   8,  0.0960),
    ("AWS",   "m6i.xlarge",    4,  16,  0.1920),
    ("AWS",   "m6i.2xlarge",   8,  32,  0.3840),
    ("AWS",   "m6i.4xlarge",  16,  64,  0.7680),
    # Azure (B / D-series general purpose)
    ("Azure", "B2ms",          2,   8,  0.0832),
    ("Azure", "D2s_v5",        2,   8,  0.1000),
    ("Azure", "D4s_v5",        4,  16,  0.2000),
    ("Azure", "D8s_v5",        8,  32,  0.4000),
    ("Azure", "D16s_v5",      16,  64,  0.8000),
    # GCP   (e2 / n2 general purpose)
    ("GCP",   "e2-medium",     2,   4,  0.0335),
    ("GCP",   "e2-standard-2", 2,   8,  0.0670),
    ("GCP",   "n2-standard-2", 2,   8,  0.0971),
    ("GCP",   "n2-standard-4", 4,  16,  0.1942),
    ("GCP",   "n2-standard-8", 8,  32,  0.3884),
    ("GCP",   "n2-standard-16",16, 64,  0.7768),
]

CATALOGUE = pd.DataFrame(
    CATALOGUE,
    columns=["provider", "instance_type", "vcpu", "ram_gb", "price_hr"],
)

# Regions per provider with a cost multiplier relative to the cheapest region.
REGIONS = {
    "AWS":   {"us-east-1": 1.00, "eu-west-2": 1.08, "ap-southeast-1": 1.12},
    "Azure": {"eastus": 1.00, "uksouth": 1.07, "southeastasia": 1.13},
    "GCP":   {"us-central1": 1.00, "europe-west2": 1.09, "asia-southeast1": 1.11},
}

# Per-GB monthly storage prices (block storage, SSD).
STORAGE_PRICE = {"AWS": 0.080, "Azure": 0.075, "GCP": 0.078}

# Managed database flat monthly add-on if enabled (small managed instance).
MANAGED_DB_PRICE = {"AWS": 58.0, "Azure": 55.0, "GCP": 60.0}

# Load balancer flat monthly add-on if enabled.
LB_PRICE = {"AWS": 18.0, "Azure": 16.0, "GCP": 18.5}


def egress_cost(provider, gb):
    """Tiered egress pricing: first 100 GB free, then two tiers."""
    base = {"AWS": 0.09, "Azure": 0.087, "GCP": 0.085}[provider]
    free = 100.0
    billable = max(0.0, gb - free)
    tier1 = min(billable, 10_000.0) * base
    tier2 = max(0.0, billable - 10_000.0) * base * 0.85  # volume discount
    return tier1 + tier2


rows = []
for _ in range(N):
    cat = CATALOGUE.sample(1, random_state=int(RNG.integers(0, 1_000_000))).iloc[0]
    provider = cat["provider"]
    region = RNG.choice(list(REGIONS[provider].keys()))
    region_mult = REGIONS[provider][region]

    instance_count = int(RNG.integers(1, 21))          # 1..20 instances
    hours = float(RNG.uniform(180, 744))               # monthly running hours
    storage_gb = float(RNG.uniform(20, 4000))
    egress_gb = float(RNG.uniform(0, 20000))
    managed_db = int(RNG.random() < 0.45)
    load_balancer = int(RNG.random() < 0.55)
    avg_cpu_util = float(RNG.uniform(5, 95))           # noise feature

    compute = cat["price_hr"] * hours * instance_count * region_mult
    storage = STORAGE_PRICE[provider] * storage_gb * region_mult
    egress = egress_cost(provider, egress_gb)
    db = MANAGED_DB_PRICE[provider] * managed_db * region_mult
    lb = LB_PRICE[provider] * load_balancer

    total = compute + storage + egress + db + lb
    # small multiplicative measurement noise
    total *= RNG.normal(1.0, 0.015)

    rows.append({
        "provider": provider,
        "region": region,
        "instance_type": cat["instance_type"],
        "vcpu": int(cat["vcpu"]),
        "ram_gb": int(cat["ram_gb"]),
        "instance_count": instance_count,
        "monthly_hours": round(hours, 1),
        "storage_gb": round(storage_gb, 1),
        "egress_gb": round(egress_gb, 1),
        "managed_db": managed_db,
        "load_balancer": load_balancer,
        "avg_cpu_util": round(avg_cpu_util, 1),
        "monthly_cost": round(total, 2),
    })

df = pd.DataFrame(rows)

# Write dataset to project-relative data/ directory
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
outpath = DATA_DIR / "cloud_costs.csv"
df.to_csv(outpath, index=False)
print(f"Wrote {len(df)} rows to {outpath}")
print(df.describe(include="all").T[["count", "mean", "min", "max"]])
print("\nClass/provider balance:\n", df["provider"].value_counts())
print("\nCost summary: mean ${:.2f}  min ${:.2f}  max ${:.2f}".format(
    df["monthly_cost"].mean(), df["monthly_cost"].min(), df["monthly_cost"].max()))
