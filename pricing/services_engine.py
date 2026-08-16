"""
services_engine.py
-------------------
Python mirror of the pricing math implemented client-side in
backend/static/index.html (planningCost / serviceCost / providerTotal /
pickRecommended). Both read the same source of truth,
pricing/services_catalog.json, so a scenario priced through the
POST /api/scenario endpoint matches what the dashboard shows.

This is what makes /api/scenario usable from scripts, CI/CD pipelines or any
other automation: a scenario JSON in, a full multi-cloud cost breakdown out.
"""

import json
from pathlib import Path

from . import cost_engine

CATALOG_PATH = Path(__file__).with_name("services_catalog.json")
PROVIDERS = ["AWS", "Azure", "GCP"]
PM_MULT = {"ondemand": 1.0, "reserved": 0.62, "spot": 0.30}
DEFAULT_WORKLOAD = {"min_vcpu": 4, "min_ram_gb": 16, "instance_count": 4, "monthly_hours": 730}
DEFAULT_REGIONS = {"AWS": "eu-west-2", "Azure": "uksouth", "GCP": "europe-west2"}


def load_services():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {s["id"]: s for s in data["services"]}, data["services"]


def _planning_state(service, overrides):
    """Field values for one service: request overrides layered on the catalogue defaults."""
    state = {}
    for field in (service.get("planning") or {}).get("fields", []):
        state[field["id"]] = field["default"]
    state.update(overrides or {})
    return state


def planning_cost(service, provider, overrides):
    rates = service.get("rates") or {}
    base = rates.get(provider)
    if base is None:
        return None
    state = _planning_state(service, overrides)
    total = base
    for field in (service.get("planning") or {}).get("fields", []):
        value = state.get(field["id"], field["default"])
        if field["type"] == "select":
            opt = next((o for o in field.get("options", []) if o["value"] == value),
                       (field.get("options") or [{}])[0])
            total *= opt.get("mult", 1) if opt else 1
        elif field["type"] == "number" and field.get("role") in ("count", "perunit"):
            try:
                total *= float(value)
            except (TypeError, ValueError):
                total *= 0
    return round(total, 2)


def _vm_cost(catalog, provider, region, workload, pricing_model):
    """Cheapest viable instance for a provider/region; falls back to the provider's
    default region if `region` isn't present in the server-side catalogue (the
    dashboard's expanded region list is currently a client-side-only demo dataset)."""
    regions_available = catalog["providers"][provider]["regions"]
    used_region = region if region in regions_available else DEFAULT_REGIONS[provider]
    rates = regions_available[used_region]
    viable = [i for i in rates["instances"]
              if i["vcpu"] >= workload["min_vcpu"] and i["ram_gb"] >= workload["min_ram_gb"]]
    if not viable:
        return None
    mult = PM_MULT.get(pricing_model, 1.0)
    best = min(viable, key=lambda i: i["price_hr"])
    compute = best["price_hr"] * workload["monthly_hours"] * workload["instance_count"] * mult
    return {
        "instance_type": best["instance_type"], "vcpu": best["vcpu"], "ram_gb": best["ram_gb"],
        "price_hr": best["price_hr"], "region_used": used_region,
        "region_label": rates.get("label", used_region), "region_fallback": used_region != region,
        "vcpu_total": best["vcpu"] * workload["instance_count"],
        "monthly_cost": round(compute, 2),
    }


def price_scenario(scenario, catalog=None):
    """Price a scenario the same way the dashboard does.

    scenario = {
      "selected": ["vm", "object_storage", ...],
      "workload": {"min_vcpu":.., "min_ram_gb":.., "instance_count":.., "monthly_hours":..},
      "regions": {"AWS":.., "Azure":.., "GCP":..},
      "pricing_model": "ondemand" | "reserved" | "spot",
      "objective": "cost" | "value",
      "budget": 0,
      "planning_state": {"<service_id>": {"<field_id>": value, ...}, ...}
    }
    """
    catalog = catalog or cost_engine.load_catalog()
    services_by_id, _ = load_services()

    selected_ids = [sid for sid in (scenario.get("selected") or []) if sid in services_by_id]
    workload = {**DEFAULT_WORKLOAD, **(scenario.get("workload") or {})}
    regions = {**DEFAULT_REGIONS, **(scenario.get("regions") or {})}
    pricing_model = scenario.get("pricing_model", "ondemand")
    objective = scenario.get("objective", "cost")
    budget = float(scenario.get("budget") or 0)
    planning_state = scenario.get("planning_state") or {}

    vm_detail = {}
    services_out = []
    totals = {p: 0.0 for p in PROVIDERS}
    for sid in selected_ids:
        svc = services_by_id[sid]
        row = {"id": sid, "category": svc["category"], "group": svc["group"],
               "model": svc["model"], "products": svc["products"], "cost": {}}
        for p in PROVIDERS:
            if svc["model"] == "vm":
                if p not in vm_detail:
                    vm_detail[p] = _vm_cost(catalog, p, regions[p], workload, pricing_model)
                cost = vm_detail[p]["monthly_cost"] if vm_detail[p] else None
            else:
                cost = planning_cost(svc, p, planning_state.get(sid))
            row["cost"][p] = cost
            if cost is not None:
                totals[p] += cost
        services_out.append(row)

    for p in PROVIDERS:
        totals[p] = round(totals[p], 2)

    has_vm = any(services_by_id[sid]["model"] == "vm" for sid in selected_ids)
    perf_units = {p: (vm_detail.get(p) or {}).get("vcpu_total", 0) if has_vm else 1 for p in PROVIDERS}
    efficiency = {p: (round(totals[p] / perf_units[p], 4) if perf_units[p] else None) for p in PROVIDERS}

    pool = PROVIDERS
    if budget > 0:
        within = [p for p in PROVIDERS if totals[p] <= budget]
        if within:
            pool = within
    if objective == "value" and all(efficiency[p] is not None for p in pool):
        recommended = min(pool, key=lambda p: efficiency[p])
    else:
        recommended = min(pool, key=lambda p: totals[p]) if selected_ids else None
    best_value = min((p for p in PROVIDERS if efficiency[p] is not None),
                      key=lambda p: efficiency[p], default=None)

    return {
        "selected": selected_ids, "workload": workload, "regions": regions,
        "pricing_model": pricing_model, "objective": objective, "budget": budget,
        "services": services_out, "totals": totals, "efficiency": efficiency,
        "vm_instances": vm_detail if has_vm else None,
        "recommended_provider": recommended, "best_value_provider": best_value,
        "over_budget": {p: (budget > 0 and totals[p] > budget) for p in PROVIDERS},
    }
