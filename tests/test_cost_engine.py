import json
from pathlib import Path

from pricing import cost_engine


def test_best_for_provider_loads_catalog():
    cat = cost_engine.load_catalog()
    # exercise best_for_provider for AWS eu-west-2 with a small workload
    wl = {"min_vcpu": 2, "min_ram_gb": 4, "instance_count": 1, "monthly_hours": 100,
          "storage_gb": 10, "egress_gb": 50, "managed_db": 0, "load_balancer": 0}
    best = cost_engine.best_for_provider(cat, "AWS", "eu-west-2", wl)
    assert best is not None
    assert "monthly_cost" in best
    assert best["monthly_cost"] >= 0


def test_load_catalog_creates_default_when_missing(tmp_path):
    missing = tmp_path / "catalog_cache.json"
    cat = cost_engine.load_catalog(missing)
    assert missing.exists()
    assert cat["currency"] == "USD"
    assert set(cat["providers"]).issuperset({"AWS", "Azure", "GCP"})
    assert cat["providers"]["AWS"]["regions"]["eu-west-2"]["instances"]


def test_compare_providers_accepts_region_aliases():
    cat = cost_engine.load_catalog()
    wl = {"min_vcpu": 4, "min_ram_gb": 16, "instance_count": 4, "monthly_hours": 730,
          "storage_gb": 500, "egress_gb": 1500, "managed_db": 1, "load_balancer": 1}
    result = cost_engine.compare_providers(cat, wl, {
        "AWS": "us-east-1",
        "Azure": "eastus2",
        "GCP": "europe-west1",
    })
    assert "recommended" in result
    assert len(result["providers"]) == 3
    assert {p["provider"] for p in result["providers"]} == {"AWS", "Azure", "GCP"}


def test_services_catalog_file_exists_and_has_data():
    catalog_path = Path(__file__).resolve().parents[1] / "pricing" / "services_catalog.json"
    assert catalog_path.exists(), "services catalog file is missing"
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert isinstance(data.get("services"), list)
    assert data["services"]
