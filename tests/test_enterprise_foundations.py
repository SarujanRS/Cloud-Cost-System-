from backend.app import app


def test_enterprise_summary_reports_database_and_security_baseline():
    client = app.test_client()
    response = client.get("/api/enterprise/summary")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["database"]["engine"] in {"postgresql", "mysql", "sqlite"}
    assert payload["security"]["https_required"] is False or isinstance(payload["security"]["https_required"], bool)
    assert "monitoring" in payload
    assert "budget_controls" in payload
