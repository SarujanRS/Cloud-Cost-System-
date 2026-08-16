import json
from pathlib import Path

import joblib
import pandas as pd

from backend.app import app

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "ml" / "model.joblib"
METRICS_PATH = ROOT / "ml" / "metrics.json"


def test_ml_artifacts_exist_and_include_results():
    assert MODEL_PATH.exists(), "Expected trained model artifact to exist in ml/model.joblib"
    assert METRICS_PATH.exists(), "Expected metrics artifact to exist in ml/metrics.json"

    with METRICS_PATH.open() as fh:
        metrics = json.load(fh)

    assert "results" in metrics
    assert "best" in metrics
    assert metrics["best"] in metrics["results"]


def test_ml_model_predicts_positive_cost_for_sample_row():
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([
        {
            "provider": "AWS",
            "region": "eu-west-2",
            "instance_type": "t3.large",
            "vcpu": 2,
            "ram_gb": 8,
            "instance_count": 2,
            "monthly_hours": 720,
            "storage_gb": 200,
            "egress_gb": 500,
            "managed_db": 0,
            "load_balancer": 0,
            "avg_cpu_util": 60,
        }
    ])
    pred = float(model.predict(sample)[0])
    assert pred > 0


def test_api_providers_endpoint_returns_pricing_snapshot():
    client = app.test_client()
    response = client.get("/api/providers")
    assert response.status_code == 200
    payload = response.get_json()
    assert "providers" in payload
    assert set(payload["providers"]).issuperset({"AWS", "Azure", "GCP"})


def test_api_compare_endpoint_returns_recommendation_data():
    client = app.test_client()
    payload = {
        "min_vcpu": 2,
        "min_ram_gb": 4,
        "instance_count": 2,
        "monthly_hours": 720,
        "storage_gb": 200,
        "egress_gb": 500,
        "managed_db": 0,
        "load_balancer": 0,
        "avg_cpu_util": 60,
    }
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    result = response.get_json()
    assert "providers" in result
    assert "recommended" in result
    assert result["recommended"]["monthly_cost"] > 0


def test_api_predict_endpoint_uses_model_if_available():
    client = app.test_client()
    response = client.post(
        "/api/predict",
        json={
            "provider": "AWS",
            "region": "eu-west-2",
            "instance_type": "t3.large",
            "vcpu": 2,
            "ram_gb": 8,
            "instance_count": 2,
            "monthly_hours": 720,
            "storage_gb": 200,
            "egress_gb": 500,
            "managed_db": 0,
            "load_balancer": 0,
            "avg_cpu_util": 60,
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert "predicted_monthly_cost" in payload
    assert payload["predicted_monthly_cost"] > 0


def test_ai_planner_endpoint_returns_workload_guidance():
    client = app.test_client()
    response = client.post(
        "/api/ai-assist",
        json={
            "prompt": "I want to build a web app with a SQL database and storage for a small business in Azure",
            "selected_services": ["vm", "object_storage", "managed_db"],
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert "recommendation" in payload
    assert "matched_services" in payload
    assert "preferred_provider" in payload
    assert payload["preferred_provider"] == "Azure"


def test_ai_planner_requests_follow_up_when_prompt_is_not_specific_enough():
    client = app.test_client()
    response = client.post(
        "/api/ai-assist",
        json={
            "prompt": "I need an app",
            "selected_services": [],
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload.get("needs_follow_up") is True
    assert isinstance(payload.get("next_question"), str)
    assert payload.get("next_question")


def test_frontend_uses_response_aware_forecast_warning_message():
    html = (ROOT / "backend" / "static" / "index.html").read_text(encoding="utf-8")
    assert "Forecast unavailable — the backend ML model isn\'t reachable right now." not in html
    assert "Forecast unavailable" in html
