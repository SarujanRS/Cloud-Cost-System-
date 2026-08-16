from backend.app import app


def test_enterprise_budget_and_chargeback_flow():
    client = app.test_client()

    org = client.post("/api/enterprise/organizations", json={"name": "Contoso"}).get_json()
    team = client.post("/api/enterprise/teams", json={"organization_id": org["id"], "name": "Platform"}).get_json()

    assert org["name"] == "Contoso"
    assert team["name"] == "Platform"

    tag = client.post(
        "/api/enterprise/tags",
        json={
            "organization_id": org["id"],
            "entity_type": "team",
            "entity_id": team["id"],
            "key": "owner",
            "value": "engineering",
        },
    ).get_json()
    assert tag["value"] == "engineering"

    budget = client.post(
        "/api/enterprise/budgets",
        json={
            "organization_id": org["id"],
            "team_id": team["id"],
            "monthly_limit": 1000,
            "alert_threshold": 0.8,
        },
    ).get_json()
    assert budget["monthly_limit"] == 1000

    check = client.post(
        "/api/enterprise/budgets/check",
        json={
            "budget_id": budget["id"],
            "actual_cost": 850,
            "month": "2026-08",
        },
    ).get_json()
    assert check["status"] in {"ok", "warning", "critical"}
    assert "alert" in check

    chargeback = client.get(f"/api/enterprise/chargeback?organization_id={org['id']}&month=2026-08").get_json()
    assert "items" in chargeback
    assert isinstance(chargeback["items"], list)
