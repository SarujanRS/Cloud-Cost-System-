"""
billing_import.py
------------------
Parses a simplified, common billing CSV format (date, provider, service,
region, amount) so actual spend can be reconciled against a scenario's
estimate. This is deliberately NOT the official AWS Cost and Usage Report /
Azure Cost Management export / GCP BigQuery billing export schema - those
are large, provider-specific and change over time. This covers the common
subset (what was spent, by which provider/service, when) that a
reconciliation view actually needs.
"""

import csv
import io
import statistics

REQUIRED_COLUMNS = {"date", "provider", "service", "amount"}
PROVIDER_ALIASES = {
    "AWS": "AWS", "AMAZON": "AWS", "AMAZON WEB SERVICES": "AWS",
    "AZURE": "Azure", "MICROSOFT": "Azure", "MICROSOFT AZURE": "Azure",
    "GCP": "GCP", "GOOGLE": "GCP", "GOOGLE CLOUD": "GCP", "GOOGLE CLOUD PLATFORM": "GCP",
}
TEMPLATE_CSV = (
    "date,provider,service,region,amount\n"
    "2026-07-01,AWS,Amazon EC2,us-east-1,412.50\n"
    "2026-07-01,Azure,Azure Virtual Machines,eastus,398.10\n"
    "2026-07-01,GCP,Compute Engine,us-central1,375.20\n"
    "2026-07-01,AWS,Amazon S3,us-east-1,18.40\n"
    "2026-07-01,Azure,Azure Blob Storage,eastus,15.90\n"
)


def parse_billing_csv(file_stream):
    """Parse an uploaded CSV file-like object into normalised row dicts.
    Raises ValueError with a human-readable message on any format problem."""
    raw = file_stream.read()
    text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("the file looks empty or isn't a valid CSV")

    headers = {(h or "").strip().lower() for h in reader.fieldnames}
    missing = REQUIRED_COLUMNS - headers
    if missing:
        raise ValueError(
            "missing required column(s): " + ", ".join(sorted(missing))
            + ". Expected headers: date, provider, service, region (optional), amount."
        )

    rows = []
    for line_no, raw_row in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw_row.items()}
        if not any(row.values()):
            continue  # skip blank lines
        provider = PROVIDER_ALIASES.get(row.get("provider", "").upper(), row.get("provider", "").strip() or "Other")
        try:
            amount = float(row.get("amount") or 0)
        except ValueError:
            raise ValueError(f"row {line_no}: '{row.get('amount')}' is not a valid amount")
        rows.append({
            "date": row.get("date", ""), "provider": provider,
            "service": row.get("service", "") or "Unclassified",
            "region": row.get("region", ""), "amount": amount,
        })

    if not rows:
        raise ValueError("no data rows found in the file")
    return rows


def summarize_billing(rows):
    totals = {"AWS": 0.0, "Azure": 0.0, "GCP": 0.0, "Other": 0.0}
    by_service = {}
    by_date = {}
    dates = [r["date"] for r in rows if r["date"]]
    for r in rows:
        p = r["provider"] if r["provider"] in ("AWS", "Azure", "GCP") else "Other"
        totals[p] += r["amount"]
        bucket = by_service.setdefault(r["service"], {"AWS": 0.0, "Azure": 0.0, "GCP": 0.0, "Other": 0.0})
        bucket[p] += r["amount"]
        if r["date"]:
            day = by_date.setdefault(r["date"], {"AWS": 0.0, "Azure": 0.0, "GCP": 0.0, "Other": 0.0})
            day[p] += r["amount"]

    for p in totals:
        totals[p] = round(totals[p], 2)
    for svc in by_service:
        for p in by_service[svc]:
            by_service[svc][p] = round(by_service[svc][p], 2)

    # Sorted daily series (date + per-provider total) so the UI can chart actual
    # spend over time, or roll it up into weekly/monthly buckets client-side.
    daily = []
    for date in sorted(by_date):
        row = {"date": date}
        for p in by_date[date]:
            row[p] = round(by_date[date][p], 2)
        daily.append(row)

    return {
        "totals": totals, "by_service": by_service, "row_count": len(rows),
        "date_range": {"from": min(dates), "to": max(dates)} if dates else None,
        "daily": daily,
    }


def detect_anomalies(rows, z_threshold=3.5, min_days=4):
    """Flag provider-days whose spend is a statistical outlier against that
    same provider's own history in this file.

    Uses the median/MAD-based "modified z-score" (Iglewicz & Hoya), not a
    plain mean/stdev z-score: with only a handful of data points, a single
    extreme spike inflates the standard deviation enough to mask itself (the
    "masking effect") - a naive z-score can miss the exact outlier it's
    supposed to catch. Median and MAD are far more robust to that. Threshold
    3.5 is the commonly cited default for the modified z-score (it is NOT
    on the same scale as an ordinary z-score, so don't compare the two).

    Needs at least `min_days` distinct dates of history for a provider before
    it will flag anything; with fewer points there isn't enough history to
    call anything an outlier."""
    daily = {}  # (provider, date) -> total
    for r in rows:
        if not r["date"]:
            continue
        p = r["provider"] if r["provider"] in ("AWS", "Azure", "GCP") else "Other"
        key = (p, r["date"])
        daily[key] = daily.get(key, 0.0) + r["amount"]

    by_provider = {}
    for (p, date), amount in daily.items():
        by_provider.setdefault(p, []).append((date, amount))

    anomalies = []
    for p, points in by_provider.items():
        if len(points) < min_days:
            continue
        amounts = [a for _, a in points]
        median = statistics.median(amounts)
        mad = statistics.median([abs(a - median) for a in amounts])
        if mad == 0:
            continue  # every day within the median is identical - nothing to flag
        for date, amount in points:
            z = 0.6745 * (amount - median) / mad
            if abs(z) >= z_threshold:
                anomalies.append({
                    "provider": p, "date": date, "amount": round(amount, 2),
                    "baseline": round(median, 2), "z_score": round(z, 2),
                    "direction": "spike" if z > 0 else "drop",
                    "drivers": _service_root_cause(rows, p, date),
                })
    anomalies.sort(key=lambda a: -abs(a["z_score"]))
    return anomalies


def _service_root_cause(rows, provider, date):
    """For a flagged (provider, date) anomaly, name the service(s) that most
    drove the deviation: each service's spend on that day vs its own median
    spend (across every day it appears for this provider). Not a claim of
    exact causal attribution - just the biggest contributor(s) to the day
    looking unusual, which is what "why did this spike" actually needs."""
    by_service_day = {}
    for r in rows:
        if not r["date"]:
            continue
        p = r["provider"] if r["provider"] in ("AWS", "Azure", "GCP") else "Other"
        if p != provider:
            continue
        by_service_day.setdefault(r["service"], {}).setdefault(r["date"], 0.0)
        by_service_day[r["service"]][r["date"]] += r["amount"]

    contributions = []
    for service, day_map in by_service_day.items():
        if date not in day_map:
            continue
        amounts = list(day_map.values())
        median = statistics.median(amounts) if len(amounts) >= 2 else 0.0
        delta = day_map[date] - median
        contributions.append({
            "service": service, "amount": round(day_map[date], 2),
            "baseline": round(median, 2), "delta": round(delta, 2),
        })
    contributions.sort(key=lambda c: -abs(c["delta"]))
    return contributions[:3]
