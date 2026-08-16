"""
app.py - Flask REST API for the Multi-Cloud Cost Console.

Endpoints
    GET  /                     the console dashboard
    GET  /api/providers        live on-demand pricing per provider (+ source/latency)
    POST /api/compare          three-cloud cost comparison for a workload
    POST /api/predict          ML-predicted monthly cost for one configuration
    POST /api/forecast         same as /api/compare, plus an 80% confidence band per
                                provider from the Random Forest's tree-level spread
    POST /api/refresh          refresh the in-memory catalogue from live pricing APIs
    GET  /api/metrics          model evaluation metrics from the last training run
    POST /api/train            on-demand retrain (fast, no k-fold CV) with fresh metrics
    GET  /api/services         the cross-provider service catalogue (source of truth)
    POST /api/scenario         price a full scenario (selected services + workload +
                                regions + pricing model) - the automation/CI-CD endpoint
    POST /api/share            persist a scenario for a shareable link + comment thread
    GET  /api/share/<id>       fetch a shared scenario and its comments
    POST /api/share/<id>/comments   add a comment to a shared scenario
    GET  /api/billing/template  sample CSV for the actual-billing import format
    POST /api/billing/import   parse an actual-billing CSV, summarise it for
                                estimate-vs-actual reconciliation, and flag
                                statistical spend anomalies (z-score per provider)
    GET  /manifest.json, /icon.svg, /sw.js   PWA manifest, icon and service worker
    POST /api/auth/register, /api/auth/login, /api/auth/logout   account creation & sessions
    GET  /api/auth/me         the signed-in user, if any
    POST /api/auth/profile    update display name / role
    POST /api/auth/password   change password
    GET  /api/audit           recent audit trail entries (Owner only)
    GET  /api/audit/export    audit trail as a CSV download (Owner only)

The rule-based cost comes straight from live (or cached) rates and is fully
transparent; the ML model gives a separate predicted figure for comparison.
Collaboration (shares/comments) has no authentication - see pricing/collab_store.py.
Accounts (pricing/auth_store.py) use real hashed passwords + Flask sessions. Session
cookies are HttpOnly/SameSite=Lax with a persisted signing key (see
_load_or_create_secret_key below); the Secure flag and actual TLS are opt-in via
FLASK_HTTPS=1 (self-signed, for local testing) or a real reverse proxy + FLASK_COOKIE_
SECURE=1 in production - see the bottom of this file and app.config.update() above.

RBAC is enforced server-side, not just hidden in the UI: require_role() (below)
re-reads the caller's role from the database on every call and rejects with
401/403, currently applied to /api/refresh and /api/train (Editor+ - they mutate
shared, non-per-user state) and /api/audit* (Owner only). Account-scoped actions
(profile/password) just require being signed in as that account. Every account
and role-affecting action is written to an audit trail - see pricing/audit_store.py.
There's no team/org model yet, so a user's role is currently self-service (Settings
> Role) rather than admin-assigned; real cross-user role assignment needs the
multi-tenant/org model that's explicitly out of scope for now.

/api/auth/login is rate-limited (pricing/rate_limit.py): 5 failed attempts per
username or 20 per source IP within 15 minutes locks that username/IP out for
15 minutes (429), independent in-memory sliding windows, reset per-username on
a successful login. Failed attempts are also written to the audit trail as
login_failed so an Owner can see brute-force activity, not just successes.
"""

import os
import sys
import json
import copy
import secrets
import subprocess
import datetime as dt
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, Response, session, g

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pricing import providers, cost_engine, services_engine, collab_store, billing_import, auth_store, audit_store, rate_limit
from pricing.database import get_database_status

app = Flask(__name__, static_folder="static")


def _load_or_create_secret_key():
    """Session signing key: FLASK_SECRET_KEY env var wins; otherwise a key persisted
    to pricing/secret.key so sessions survive a server restart. To force everyone to
    re-login (e.g. after a suspected compromise), delete that file or set the env var.
    """
    env_key = os.environ.get("FLASK_SECRET_KEY")
    if env_key:
        return env_key
    key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pricing", "secret.key")
    if os.path.exists(key_path):
        with open(key_path, "r", encoding="utf-8") as f:
            existing = f.read().strip()
        if existing:
            return existing
    new_key = secrets.token_hex(32)
    with open(key_path, "w", encoding="utf-8") as f:
        f.write(new_key)
    try:
        os.chmod(key_path, 0o600)  # best-effort; no-op on filesystems that ignore POSIX perms
    except OSError:
        pass
    return new_key


app.secret_key = _load_or_create_secret_key()

# Session cookie hardening. SECURE is off by default because the plain `python app.py`
# dev server speaks HTTP, and browsers silently drop Secure cookies over HTTP (you'd
# get logged out, not a security warning). Set FLASK_HTTPS=1 (see bottom of this file)
# or put a real TLS-terminating reverse proxy in front of this app, then set
# FLASK_COOKIE_SECURE=1 so the cookie is only ever sent over HTTPS.
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_COOKIE_SECURE", "0") == "1",
    PERMANENT_SESSION_LIFETIME=dt.timedelta(days=14),
)

# In-memory catalogue: starts from the cache, updated in place by /api/refresh.
CATALOG = cost_engine.load_catalog()
LAST_REFRESH = {"at": CATALOG.get("updated"), "sources": {}}

# --- RBAC enforcement -------------------------------------------------------
# Roles are stored server-side per account (pricing/auth_store.py); this is the
# only place that actually restricts an API call by role. Everywhere else
# (applyRoleGatingStatic() in the frontend) just disables buttons for UX - that
# alone is not a security boundary, which is why this exists.
ROLE_RANK = {"Viewer": 1, "Editor": 2, "Owner": 3}


def current_session_user():
    uid = session.get("user_id")
    return auth_store.get_user(uid) if uid else None


def require_role(min_role):
    """Reject the request unless the signed-in user's role (read fresh from the
    database, never trusted from the client) meets or exceeds min_role."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = current_session_user()
            if not user:
                return jsonify({"error": "Sign in required"}), 401
            if ROLE_RANK.get(user["role"], 0) < ROLE_RANK[min_role]:
                return jsonify({"error": f"Requires {min_role} role or higher (you are {user['role']})"}), 403
            g.current_user = user
            return fn(*args, **kwargs)
        return wrapper
    return decorator

# ML model (optional): predicted cost for comparison with the live rule-based cost.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "model.joblib")
MODEL = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
# Random Forest, kept alongside the primary (best) model purely so /api/forecast can
# derive a confidence band from the spread across its individual trees - the primary
# model (usually Gradient Boosting) doesn't expose per-estimator predictions this way.
RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "model_rf.joblib")
RF_MODEL = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
ML_FEATURES = ["provider", "region", "instance_type", "vcpu", "ram_gb",
               "instance_count", "monthly_hours", "storage_gb", "egress_gb",
               "managed_db", "load_balancer", "avg_cpu_util"]

DEFAULT_REGIONS = {"AWS": "eu-west-2", "Azure": "uksouth", "GCP": "europe-west2"}



def _merge_live(fetch_result):
    """Merge a providers.fetch_all() result into the in-memory catalogue."""
    for prov in ["AWS", "Azure", "GCP"]:
        res = fetch_result[prov]
        region = res["region"]
        node = CATALOG["providers"][prov]["regions"].get(region)
        if node is not None:
            node["instances"] = res["instances"]
        CATALOG["providers"][prov]["source"] = res["source"]
        LAST_REFRESH["sources"][prov] = {
            "source": res["source"], "region": region,
            "latency_ms": res.get("latency_ms"), "error": res.get("error"),
        }
    LAST_REFRESH["at"] = fetch_result["fetched_at"]
    CATALOG["updated"] = fetch_result["fetched_at"]


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "app": "multicloud-cost-system", "timestamp": dt.datetime.utcnow().isoformat() + "Z"})


@app.route("/ready", methods=["GET"])
def readiness_check():
    ready = bool(CATALOG and "providers" in CATALOG)
    return jsonify({"status": "ok" if ready else "degraded", "ready": ready, "catalog_loaded": ready, "model_loaded": MODEL is not None})


@app.route("/api/providers", methods=["GET"])
def api_providers():
    """Current pricing per provider for the active regions, with data source."""
    out = {"updated": CATALOG.get("updated"), "currency": CATALOG.get("currency"),
           "providers": {}}
    for prov in ["AWS", "Azure", "GCP"]:
        region = DEFAULT_REGIONS[prov]
        node = CATALOG["providers"][prov]["regions"][region]
        out["providers"][prov] = {
            "api": CATALOG["providers"][prov].get("api"),
            "source": CATALOG["providers"][prov].get("source", "cache"),
            "region": region, "region_label": node.get("label", region),
            "instances": node["instances"],
            "storage_gb_month": node["storage_gb_month"],
            "egress_gb": node["egress_gb"],
            "managed_db_month": node["managed_db_month"],
            "load_balancer_month": node["load_balancer_month"],
        }
    out["last_refresh"] = LAST_REFRESH
    return jsonify(out)


@app.route("/api/refresh", methods=["POST"])
@require_role("Editor")
def api_refresh():
    """Fetch live prices from all three providers and update the catalogue.
    Editor+ only - this changes the shared in-memory catalogue for everyone."""
    regions = (request.get_json(silent=True) or {}).get("regions", DEFAULT_REGIONS)
    result = providers.fetch_all(regions)
    _merge_live(result)
    audit_store.log_event(g.current_user["id"], g.current_user["username"], "catalog_refresh",
                           f"regions={regions}")
    return jsonify({"ok": True, "last_refresh": LAST_REFRESH,
                    "updated": CATALOG["updated"]})


def _ml_row(provider, region, best, workload):
    return {"provider": provider, "region": region,
            "instance_type": best["instance_type"], "vcpu": best["vcpu"],
            "ram_gb": best["ram_gb"], "instance_count": workload["instance_count"],
            "monthly_hours": workload["monthly_hours"],
            "storage_gb": workload["storage_gb"], "egress_gb": workload["egress_gb"],
            "managed_db": workload.get("managed_db", 0),
            "load_balancer": workload.get("load_balancer", 0),
            "avg_cpu_util": workload.get("avg_cpu_util", 60)}


def _ml_predict(provider, region, best, workload):
    if MODEL is None:
        return None
    row = _ml_row(provider, region, best, workload)
    return round(float(MODEL.predict(pd.DataFrame([row])[ML_FEATURES])[0]), 2)


def _forecast_band(provider, region, best, workload):
    """80% prediction interval from the spread across the Random Forest's
    individual trees - a real (if simple) confidence band, not a fabricated one."""
    if RF_MODEL is None:
        return None
    row = pd.DataFrame([_ml_row(provider, region, best, workload)])[ML_FEATURES]
    Xt = RF_MODEL.named_steps["pre"].transform(row)
    tree_preds = np.array([tree.predict(Xt)[0] for tree in RF_MODEL.named_steps["model"].estimators_])
    return {
        "predicted": round(float(tree_preds.mean()), 2),
        "low": round(float(np.percentile(tree_preds, 10)), 2),
        "high": round(float(np.percentile(tree_preds, 90)), 2),
        "confidence": "80%",
        "source": "Random Forest tree spread (300 estimators)",
    }


_PRETTY_FEATURE = {
    "vcpu": "vCPU count", "ram_gb": "RAM (GB)", "instance_count": "Instance count",
    "monthly_hours": "Hours / month", "storage_gb": "Storage (GB)", "egress_gb": "Egress (GB)",
    "managed_db": "Managed database", "load_balancer": "Load balancer", "avg_cpu_util": "Avg CPU utilisation",
}


def _pretty_feature_name(raw):
    # sklearn ColumnTransformer names look like "cat__provider_AWS" or "num__vcpu"
    name = raw.split("__", 1)[-1]
    for cat_col in ("provider", "region", "instance_type"):
        prefix = cat_col + "_"
        if name.startswith(prefix):
            label = {"provider": "Provider", "region": "Region", "instance_type": "Instance type"}[cat_col]
            return f"{label} = {name[len(prefix):]}"
    return _PRETTY_FEATURE.get(name, name)


def _feature_importances(top_n=8):
    """What the Random Forest actually weighs most when predicting cost -
    read directly from the fitted model, not a canned explanation."""
    if RF_MODEL is None:
        return None
    pre = RF_MODEL.named_steps["pre"]
    model = RF_MODEL.named_steps["model"]
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        return None
    pairs = sorted(zip(names, model.feature_importances_), key=lambda t: t[1], reverse=True)[:top_n]
    return [{"feature": _pretty_feature_name(n), "importance": round(float(v), 4)} for n, v in pairs]


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Three-cloud cost comparison for a workload, with optional ML prediction."""
    req = request.get_json() or {}
    workload = {
        "min_vcpu": req.get("min_vcpu", 2), "min_ram_gb": req.get("min_ram_gb", 4),
        "instance_count": req.get("instance_count", 2),
        "monthly_hours": req.get("monthly_hours", 730),
        "storage_gb": req.get("storage_gb", 200), "egress_gb": req.get("egress_gb", 500),
        "managed_db": int(req.get("managed_db", 0)),
        "load_balancer": int(req.get("load_balancer", 0)),
        "avg_cpu_util": req.get("avg_cpu_util", 60),
    }
    regions = req.get("regions", DEFAULT_REGIONS)
    result = cost_engine.compare_providers(CATALOG, workload, regions)
    # attach ML predicted cost per provider
    if MODEL is not None and "providers" in result:
        for p in result["providers"]:
            p["ml_predicted_cost"] = _ml_predict(p["provider"], p["region"], p, workload)
    result["workload"] = workload
    result["updated"] = CATALOG.get("updated")
    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if MODEL is None:
        return jsonify({"error": "model not available"}), 503
    cfg = request.get_json() or {}
    row = {k: cfg.get(k) for k in ML_FEATURES}
    return jsonify({"predicted_monthly_cost":
                    round(float(MODEL.predict(pd.DataFrame([row]))[0]), 2)})


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cloud_costs.csv")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "metrics.json")
GEN_DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "generate_dataset.py")


def _load_metrics():
    """Read ml/metrics.json (written by the last training run) and fill in any
    fields older/lighter runs may not have set, so the API response shape is stable."""
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault(
        "generated_at",
        dt.datetime.utcfromtimestamp(os.path.getmtime(METRICS_PATH)).isoformat() + "Z",
    )
    if "mean_cost" not in data and os.path.exists(DATA_PATH):
        data["mean_cost"] = round(float(pd.read_csv(DATA_PATH)["monthly_cost"].mean()), 2)
    data.setdefault("cv_folds", 5 if any("CV_R2_mean" in v for v in data.get("results", {}).values()) else 0)
    return data


def _ensure_dataset():
    if not os.path.exists(DATA_PATH):
        subprocess.run([sys.executable, GEN_DATASET_PATH], check=True)


def _train_live():
    """On-demand retrain triggered from the dashboard: a single held-out test
    split (no k-fold CV) so it finishes in seconds instead of the ~1 minute the
    full offline ml/train_model.py run takes with 5-fold cross-validation."""
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    _ensure_dataset()
    df = pd.read_csv(DATA_PATH)
    y = df["monthly_cost"].values
    X = df.drop(columns=["monthly_cost"])

    cat_cols = ["provider", "region", "instance_type"]
    num_cols = ["vcpu", "ram_gb", "instance_count", "monthly_hours",
                "storage_gb", "egress_gb", "managed_db", "load_balancer", "avg_cpu_util"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.08, subsample=0.9, random_state=42),
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    results, fitted = {}, {}
    for name, est in models.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "MAE": round(float(mean_absolute_error(y_test, pred)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
            "R2": round(float(r2_score(y_test, pred)), 4),
        }
        fitted[name] = pipe

    best_name = max(results, key=lambda k: results[k]["R2"])
    best = fitted[best_name]
    joblib.dump(best, MODEL_PATH)
    joblib.dump(fitted["Random Forest"], RF_MODEL_PATH)  # kept for /api/forecast's confidence bands

    metrics = {
        "results": results, "best": best_name,
        "mean_cost": round(float(y.mean()), 2), "sample_size": int(len(df)),
        "cv_folds": 0, "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "trigger": "live-retrain",
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics, best, fitted["Random Forest"]


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Model evaluation metrics from the current ml/model.joblib training run."""
    data = _load_metrics()
    if data is None:
        return jsonify({"error": "metrics not available"}), 503
    return jsonify(data)


@app.route("/api/train", methods=["POST"])
@require_role("Editor")
def api_train():
    """Retrain on demand and hot-swap the in-memory prediction model.
    Editor+ only - this replaces the shared model for everyone."""
    global MODEL, RF_MODEL
    try:
        metrics, best, rf = _train_live()
    except Exception as e:  # dataset/training issues surface to the UI instead of a 500 page
        return jsonify({"error": str(e)}), 500
    MODEL = best
    RF_MODEL = rf
    audit_store.log_event(g.current_user["id"], g.current_user["username"], "model_retrain",
                           f"best={metrics.get('best')}")
    return jsonify(metrics)


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    """Cost forecast with an 80% confidence band per provider, alongside the
    same rule-based comparison /api/compare returns."""
    if MODEL is None:
        return jsonify({"error": "model not available"}), 503
    req = request.get_json() or {}
    workload = {
        "min_vcpu": req.get("min_vcpu", 2), "min_ram_gb": req.get("min_ram_gb", 4),
        "instance_count": req.get("instance_count", 2),
        "monthly_hours": req.get("monthly_hours", 730),
        "storage_gb": req.get("storage_gb", 200), "egress_gb": req.get("egress_gb", 500),
        "managed_db": int(req.get("managed_db", 0)),
        "load_balancer": int(req.get("load_balancer", 0)),
        "avg_cpu_util": req.get("avg_cpu_util", 60),
    }
    regions = req.get("regions", DEFAULT_REGIONS)
    result = cost_engine.compare_providers(CATALOG, workload, regions)
    if "providers" in result:
        for p in result["providers"]:
            p["ml_predicted_cost"] = _ml_predict(p["provider"], p["region"], p, workload)
            band = _forecast_band(p["provider"], p["region"], p, workload)
            if band:
                p["forecast"] = band
    result["workload"] = workload
    result["updated"] = CATALOG.get("updated")
    result["rf_available"] = RF_MODEL is not None
    result["feature_importance"] = _feature_importances()
    return jsonify(result)


@app.route("/api/services", methods=["GET"])
def api_services():
    """Return the cross-provider service catalogue used by the console."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(project_root, "pricing", "services_catalog.json")
    if not os.path.exists(path):
        return jsonify({"error": "services catalogue not found", "path": path}), 404
    with open(path, encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/api/scenario", methods=["POST"])
def api_scenario():
    """Price a full scenario programmatically - the automation/CI-CD endpoint.

    Body: {"selected": [...], "workload": {...}, "regions": {...},
           "pricing_model": "ondemand|reserved|spot", "objective": "cost|value",
           "budget": 0, "planning_state": {"<service_id>": {"<field>": value}}}
    Mirrors the pricing math the dashboard runs client-side (see
    pricing/services_engine.py), so results match what the UI shows.
    """
    scenario = request.get_json(silent=True) or {}
    try:
        result = services_engine.price_scenario(scenario, catalog=CATALOG)
    except KeyError as e:
        return jsonify({"error": f"unknown field or service id: {e}"}), 400
    result["updated"] = CATALOG.get("updated")
    return jsonify(result)


@app.route("/api/share", methods=["POST"])
def api_share_create():
    """Persist a scenario for a shareable link. No auth: anyone with the link
    can view/comment. 'role' is a display label, not an access boundary."""
    body = request.get_json(silent=True) or {}
    scenario = body.get("scenario")
    if not isinstance(scenario, dict):
        return jsonify({"error": "missing 'scenario' object"}), 400
    share_id = collab_store.create_share(scenario, body.get("owner_name"), body.get("role"))
    signed_in = current_session_user()
    audit_store.log_event(signed_in["id"] if signed_in else None, body.get("owner_name") or "Anonymous",
                           "share_created", f"share_id={share_id}")
    return jsonify({"id": share_id})


@app.route("/api/share/<share_id>", methods=["GET"])
def api_share_get(share_id):
    data = collab_store.get_share(share_id)
    if data is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(data)


@app.route("/api/share/<share_id>/comments", methods=["POST"])
def api_share_comment(share_id):
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "comment text is required"}), 400
    comments = collab_store.add_comment(share_id, body.get("author"), body.get("role"), text)
    if comments is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({"comments": comments})


@app.route("/api/billing/template", methods=["GET"])
def api_billing_template():
    return Response(billing_import.TEMPLATE_CSV, mimetype="text/csv",
                     headers={"Content-Disposition": "attachment; filename=billing_template.csv"})


@app.route("/api/billing/import", methods=["POST"])
def api_billing_import():
    """Parse an actual-billing CSV (date, provider, service, region, amount)
    and summarise it, for estimate-vs-actual reconciliation in the UI."""
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "no file uploaded (expected multipart field 'file')"}), 400
    try:
        rows = billing_import.parse_billing_csv(file.stream)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    summary = billing_import.summarize_billing(rows)
    anomalies = billing_import.detect_anomalies(rows)
    summary["anomalies"] = anomalies
    if anomalies:
        signed_in = current_session_user()
        audit_store.log_event(
            signed_in["id"] if signed_in else None, signed_in["username"] if signed_in else "anonymous",
            "billing_anomaly_detected",
            f"{len(anomalies)} anomaly(ies): " + "; ".join(
                f"{a['provider']} {a['date']} {a['direction']} (z={a['z_score']})" for a in anomalies[:5]
            ),
        )
    return jsonify(summary)


def _services_from_prompt(prompt, selected_services):
    """Map a free-form workload description to a useful set of service IDs."""
    lowercase = (prompt or "").lower()
    selected = set(selected_services or [])
    keyword_map = {
        "vm": ["web app", "website", "app", "virtual machine", "application server", "compute"],
        "object_storage": ["storage", "blob", "upload", "files", "documents", "backup"],
        "managed_db": ["sql", "database", "postgres", "mysql", "relational", "data store"],
        "nosql": ["nosql", "document", "key-value", "json", "api cache"],
        "redis": ["cache", "session", "in-memory", "fast lookups"],
        "serverless": ["serverless", "functions", "event-driven", "api"],
        "api_gateway": ["api gateway", "gateway", "public api", "integration"],
        "kubernetes": ["kubernetes", "containers", "microservices", "orchestration"],
        "apphosting": ["website hosting", "web app", "application hosting"],
        "networking": ["network", "vpn", "private connection", "vpc", "virtual network"],
        "cdn": ["cdn", "global delivery", "content delivery", "web edge"],
        "monitoring": ["monitoring", "logs", "alerts", "observability", "metrics"],
        "identity": ["identity", "iam", "auth", "users", "access"],
        "keyvault": ["secrets", "credentials", "passwords", "certificates", "keys"],
        "data_lake": ["data lake", "lakehouse", "big data", "datasets", "analytics data"],
        "datawarehouse": ["warehouse", "analytics", "reporting", "bi", "query"],
        "ml": ["ai", "machine learning", "ml", "model", "inference", "training"],
        "backup": ["backup", "disaster recovery", "restore", "nightly copy"],
        "devops": ["ci/cd", "build pipeline", "devops", "deployment", "release"],
    }

    matched = []
    for service_id, keywords in keyword_map.items():
        if service_id in selected or any(keyword in lowercase for keyword in keywords):
            matched.append(service_id)
    if not matched and selected:
        matched = list(selected)
    if not matched:
        matched = ["vm", "object_storage", "monitoring"]
    return matched


def _preferred_provider_from_prompt(prompt, matched_services):
    text = (prompt or "").lower()
    if "azure" in text:
        return "Azure"
    if "aws" in text or "amazon" in text:
        return "AWS"
    if "gcp" in text or "google cloud" in text:
        return "GCP"

    if any(s in matched_services for s in ["ml", "datawarehouse", "data_lake", "kubernetes"]):
        if "ai" in text or "analytics" in text or "model" in text:
            return "GCP"
        return "Azure"
    if any(s in matched_services for s in ["managed_db", "object_storage", "monitoring", "api_gateway"]):
        return "Azure"
    if any(s in matched_services for s in ["data_lake", "datawarehouse", "backup"]):
        return "AWS"
    if "startup" in text or "small business" in text:
        return "Azure"
    return "Azure"


def _ai_plan(prompt, selected_services):
    raw_prompt = (prompt or "").strip()
    lower = raw_prompt.lower()
    matched = _services_from_prompt(raw_prompt, selected_services)
    preferred_provider = _preferred_provider_from_prompt(raw_prompt, matched)
    region_map = {"AWS": "us-east-1", "Azure": "eastus2", "GCP": "us-east1"}

    specific_keywords = [
        "web app", "website", "api", "database", "sql", "storage", "ai", "ml",
        "analytics", "data", "microservice", "kubernetes", "serverless", "backup",
        "cache", "monitoring", "security", "identity", "mobile", "portal",
        "application server", "mobile app", "internal app"
    ]
    vague = not any(keyword in lower for keyword in specific_keywords)

    if vague:
        return {
            "needs_follow_up": True,
            "next_question": "What kind of workload are you building: web app, API, data/analytics, AI/ML, or container-based microservices?",
            "preferred_provider": preferred_provider,
            "region": region_map.get(preferred_provider, "eastus2"),
            "matched_services": matched,
            "recommendation": "I need one more detail about the workload type before I can recommend the right cloud pattern and cost model.",
            "confidence": "Medium",
        }

    if "web app" in lower or "website" in lower or "portal" in lower:
        recommendation = (
            f"Use a {preferred_provider} web workload with a VM tier, managed database, and object storage. "
            f"Prioritise {preferred_provider} regions near the user base, preferably {region_map[preferred_provider]}, "
            "and keep the architecture simple: app hosting or compute + managed DB + blob storage + monitoring."
        )
    elif "ai" in lower or "machine learning" in lower or "ml" in lower:
        recommendation = (
            f"Design for a {preferred_provider} AI workload using the compute layer for model training or inference, "
            "plus storage and monitoring. Prefer production-grade managed services, traffic routing, and a reserved compute plan "
            "if the workload is steady state."
        )
    elif "data" in lower or "analytics" in lower:
        recommendation = (
            f"Build a {preferred_provider} analytics stack with data storage, a warehouse or lakehouse layer, and monitoring. "
            "Use a cost-efficient regional deployment with a reserved compute plan for stable analytics traffic."
        )
    elif "api" in lower or "serverless" in lower or "microservice" in lower:
        recommendation = (
            f"Use a {preferred_provider} API-first design with serverless or app-hosting compute, API gateway, monitoring, and a managed database. "
            "Keep the baseline minimal and add cache, storage, and backup as traffic grows."
        )
    elif "kubernetes" in lower or "container" in lower:
        recommendation = (
            f"Deploy a {preferred_provider} container platform with Kubernetes, a managed database, object storage, and observability. "
            "Use a regional cluster near the workloads and reserve consistent nodes for lower cost."
        )
    else:
        recommendation = (
            f"Start with a lean {preferred_provider} architecture: compute + storage + monitoring, add the database and API layers "
            "only when usage grows. That keeps the baseline low-cost while leaving room for future scale."
        )

    return {
        "needs_follow_up": False,
        "preferred_provider": preferred_provider,
        "region": region_map.get(preferred_provider, "eastus2"),
        "matched_services": matched,
        "recommendation": recommendation,
        "confidence": "High" if matched else "Medium",
    }


@app.route("/api/ai-assist", methods=["POST"])
def api_ai_assist():
    """Natural-language workload planning guidance for the dashboard."""
    payload = request.get_json(silent=True) or {}
    prompt = payload.get("prompt") or ""
    selected_services = payload.get("selected_services") or []
    result = _ai_plan(prompt, selected_services)
    return jsonify(result)


@app.route("/api/enterprise/summary", methods=["GET"])
def api_enterprise_summary():
    database = get_database_status()
    return jsonify({
        "database": database,
        "security": {
            "https_required": os.environ.get("FLASK_COOKIE_SECURE", "0") == "1" or os.environ.get("FLASK_HTTPS", "0") == "1",
            "secret_key_configured": bool(os.environ.get("FLASK_SECRET_KEY")),
            "session_cookie_secure": app.config.get("SESSION_COOKIE_SECURE", False),
            "rbac_enabled": True,
            "audit_logging_enabled": True,
        },
        "monitoring": {
            "healthcheck_endpoint": "/health",
            "readiness_endpoint": "/ready",
            "error_tracking": bool(os.environ.get("SENTRY_DSN")),
            "metrics_enabled": os.environ.get("PROMETHEUS_ENABLED", "0") == "1",
            "log_level": os.environ.get("LOG_LEVEL", "INFO"),
        },
        "budget_controls": {
            "budget_threshold_enabled": bool(os.environ.get("BUDGET_ALERT_THRESHOLD")),
            "anomaly_detection_enabled": bool(os.environ.get("ANOMALY_DEVIATION_FACTOR")),
            "tagging_required": True,
            "ownership_tracking": True,
            "recommended_actions": [
                "Store spend ownership and tags in a managed database",
                "Trigger budget alerts when forecasted spend crosses threshold",
                "Review anomalous provider spend before approving new workloads",
            ],
        },
        "recommended_next_steps": [
            "Move auth and application state to PostgreSQL or MySQL",
            "Add migration scripts under migrations/versions/",
            "Run app behind TLS termination with a reverse proxy",
            "Set budget alerts and cost anomaly monitoring in production",
        ],
    })


# -----------------------------
# Enterprise feature sprint
# -----------------------------

_ORGANIZATIONS = []
_TEAMS = []
_TAGS = []
_BUDGETS = []


def _utc_now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _next_id(store):
    return max((item["id"] for item in store), default=0) + 1


@app.route("/api/enterprise/organizations", methods=["GET", "POST"])
def api_enterprise_organizations():
    if request.method == "GET":
        return jsonify({"items": _ORGANIZATIONS})

    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Organization name is required"}), 400

    org = {"id": _next_id(_ORGANIZATIONS), "name": name, "created_at": _utc_now_iso()}
    _ORGANIZATIONS.append(org)
    return jsonify(org), 201


@app.route("/api/enterprise/teams", methods=["GET", "POST"])
def api_enterprise_teams():
    if request.method == "GET":
        return jsonify({"items": _TEAMS})

    payload = request.get_json(silent=True) or {}
    org_id = payload.get("organization_id")
    name = (payload.get("name") or "").strip()
    if not org_id or not name:
        return jsonify({"error": "organization_id and name are required"}), 400

    team = {"id": _next_id(_TEAMS), "organization_id": org_id, "name": name, "created_at": _utc_now_iso()}
    _TEAMS.append(team)
    return jsonify(team), 201


@app.route("/api/enterprise/tags", methods=["GET", "POST"])
def api_enterprise_tags():
    if request.method == "GET":
        return jsonify({"items": _TAGS})

    payload = request.get_json(silent=True) or {}
    required = ["organization_id", "entity_type", "entity_id", "key", "value"]
    missing = [field for field in required if payload.get(field) in (None, "")]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    tag = {
        "id": _next_id(_TAGS),
        "organization_id": payload["organization_id"],
        "entity_type": payload["entity_type"],
        "entity_id": payload["entity_id"],
        "key": payload["key"],
        "value": payload["value"],
        "created_at": _utc_now_iso(),
    }
    _TAGS.append(tag)
    return jsonify(tag), 201


@app.route("/api/enterprise/budgets", methods=["GET", "POST"])
def api_enterprise_budgets():
    if request.method == "GET":
        return jsonify({"items": _BUDGETS})

    payload = request.get_json(silent=True) or {}
    org_id = payload.get("organization_id")
    team_id = payload.get("team_id")
    monthly_limit = payload.get("monthly_limit")
    if not org_id or team_id in (None, "") or monthly_limit is None:
        return jsonify({"error": "organization_id, team_id and monthly_limit are required"}), 400

    budget = {
        "id": _next_id(_BUDGETS),
        "organization_id": org_id,
        "team_id": team_id,
        "monthly_limit": float(monthly_limit),
        "alert_threshold": float(payload.get("alert_threshold", 0.8)),
        "currency": payload.get("currency", "USD"),
        "created_at": _utc_now_iso(),
    }
    _BUDGETS.append(budget)
    return jsonify(budget), 201


@app.route("/api/enterprise/budgets/check", methods=["POST"])
def api_enterprise_budget_check():
    payload = request.get_json(silent=True) or {}
    budget_id = payload.get("budget_id")
    actual_cost = float(payload.get("actual_cost") or 0)
    month = payload.get("month") or dt.datetime.utcnow().strftime("%Y-%m")

    budget = next((b for b in _BUDGETS if b["id"] == budget_id), None)
    if budget is None:
        return jsonify({"error": "Budget not found"}), 404

    ratio = actual_cost / float(budget["monthly_limit"]) if float(budget["monthly_limit"]) else 0
    if ratio >= 1.0:
        status = "critical"
    elif ratio >= budget["alert_threshold"]:
        status = "warning"
    else:
        status = "ok"

    result = {
        "budget_id": budget_id,
        "month": month,
        "actual_cost": actual_cost,
        "limit": float(budget["monthly_limit"]),
        "status": status,
        "alert": {
            "threshold": budget["alert_threshold"],
            "ratio": round(ratio, 4),
            "message": f"Team spend is at {round(ratio * 100, 1)}% of budget" if status != "ok" else "Spend is within budget",
        },
    }
    return jsonify(result)


@app.route("/api/enterprise/chargeback", methods=["GET"])
def api_enterprise_chargeback():
    organization_id = request.args.get("organization_id")
    month = request.args.get("month") or dt.datetime.utcnow().strftime("%Y-%m")
    items = []
    for budget in _BUDGETS:
        if organization_id and str(budget.get("organization_id")) != str(organization_id):
            continue
        items.append({
            "team_id": budget["team_id"],
            "team_name": next((team["name"] for team in _TEAMS if team["id"] == budget["team_id"]), "Unknown"),
            "limit": budget["monthly_limit"],
            "month": month,
            "allocation_percent": round((budget["alert_threshold"] * 100), 1),
        })
    return jsonify({"month": month, "items": items})


@app.route("/api/auth/register", methods=["POST"])
def api_auth_register():
    data = request.get_json(silent=True) or {}
    user_id, err = auth_store.create_user(
        data.get("username"), data.get("password"),
        display_name=data.get("display_name"), email=data.get("email"),
    )
    if err:
        return jsonify({"error": err}), 400
    session.clear()
    session["user_id"] = user_id
    session.permanent = True
    user = auth_store.get_user(user_id)
    audit_store.log_event(user_id, user["username"], "register")
    return jsonify(user)


@app.route("/api/auth/login", methods=["POST"])
def api_auth_login():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    ip = request.remote_addr or "unknown"
    wait = rate_limit.check_locked_out(username, ip)
    if wait:
        minutes = max(1, wait // 60 + (1 if wait % 60 else 0))
        return jsonify({"error": f"Too many failed attempts. Try again in about {minutes} minute(s)."}), 429
    user = auth_store.verify_user(username, data.get("password"))
    if not user:
        rate_limit.record_failure(username, ip)
        audit_store.log_event(None, (username or "").strip().lower() or "unknown", "login_failed")
        return jsonify({"error": "Incorrect username or password"}), 401
    rate_limit.record_success(username, ip)
    session.clear()
    session["user_id"] = user["id"]
    session.permanent = True
    audit_store.log_event(user["id"], user["username"], "login")
    return jsonify(user)


@app.route("/api/auth/logout", methods=["POST"])
def api_auth_logout():
    user = current_session_user()
    if user:
        audit_store.log_event(user["id"], user["username"], "logout")
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/auth/me", methods=["GET"])
def api_auth_me():
    uid = session.get("user_id")
    user = auth_store.get_user(uid) if uid else None
    if not user:
        session.clear()
        return jsonify({"error": "not authenticated"}), 401
    return jsonify(user)


@app.route("/api/auth/profile", methods=["POST"])
def api_auth_profile():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "not authenticated"}), 401
    before = auth_store.get_user(uid)
    data = request.get_json(silent=True) or {}
    user = auth_store.update_profile(uid, display_name=data.get("display_name"), role=data.get("role"))
    changes = []
    if before["display_name"] != user["display_name"]:
        changes.append(f"display_name: {before['display_name']!r} -> {user['display_name']!r}")
    if before["role"] != user["role"]:
        changes.append(f"role: {before['role']} -> {user['role']}")
    if changes:
        audit_store.log_event(uid, user["username"], "profile_update", "; ".join(changes))
    return jsonify(user)


@app.route("/api/auth/password", methods=["POST"])
def api_auth_password():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "not authenticated"}), 401
    user = auth_store.get_user(uid)
    data = request.get_json(silent=True) or {}
    ok, err = auth_store.change_password(uid, data.get("current_password"), data.get("new_password"))
    if ok:
        audit_store.log_event(uid, user["username"], "password_change")
    if not ok:
        return jsonify({"error": err}), 400
    return jsonify({"ok": True})


@app.route("/api/audit", methods=["GET"])
@require_role("Owner")
def api_audit_list():
    """Recent audit trail entries. Owner-only."""
    limit = min(int(request.args.get("limit", 200)), 1000)
    return jsonify({"events": audit_store.list_events(limit=limit)})


@app.route("/api/audit/export", methods=["GET"])
@require_role("Owner")
def api_audit_export():
    """Audit trail as a downloadable CSV. Owner-only."""
    csv_text = audit_store.export_csv()
    return Response(
        csv_text, mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit_log.csv"},
    )


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/manifest.json")
def pwa_manifest():
    return send_from_directory(app.static_folder, "manifest.json", mimetype="application/manifest+json")


@app.route("/icon.svg")
def pwa_icon():
    return send_from_directory(app.static_folder, "icon.svg", mimetype="image/svg+xml")


@app.route("/sw.js")
def service_worker():
    # Served from root (not /static/sw.js) so its default scope covers the whole
    # origin - a service worker can only control paths within its own directory.
    return send_from_directory(app.static_folder, "sw.js", mimetype="application/javascript")


if __name__ == "__main__":
    # FLASK_HTTPS=1 runs the dev server over TLS with a self-signed cert (via Werkzeug's
    # "adhoc" mode, which needs the `pyopenssl` package) so the Secure cookie flag can
    # actually be tested end to end locally. In production, terminate real TLS at a
    # reverse proxy instead and set FLASK_COOKIE_SECURE=1 rather than using this.
    use_https = os.environ.get("FLASK_HTTPS", "0") == "1"
    ssl_context = None
    if use_https:
        try:
            import OpenSSL  # noqa: F401  (presence check for the adhoc cert generator)
            ssl_context = "adhoc"
            app.config["SESSION_COOKIE_SECURE"] = True
        except ImportError:
            print("FLASK_HTTPS=1 requires the 'pyopenssl' package (pip install pyopenssl) "
                  "for self-signed dev certs. Falling back to plain HTTP.")
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=ssl_context)
