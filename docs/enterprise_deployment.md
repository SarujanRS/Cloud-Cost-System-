# Enterprise deployment checklist

## 1. Production database + migrations

- Use PostgreSQL in production unless your environment already standardizes on MySQL.
- Keep all schema and seed changes in versioned migration files under `migrations/versions/`.
- Store database credentials in environment variables and never in source control.
- Add CI/CD checks that validate migrations against a staging database before production rollout.

## 2. Secure deployment

- Run the app behind a TLS-terminating reverse proxy (nginx, Azure Front Door, or a managed ingress).
- Set `FLASK_COOKIE_SECURE=1` and only serve cookies over HTTPS.
- Use a strong `FLASK_SECRET_KEY` from a secret manager.
- Use `.env` or a managed secret store for all runtime secrets.
- Prefer a non-root container user and keep the container image minimal.

## 3. Monitoring + alerts

- Expose `/health` and `/ready` for uptime checks.
- Add Sentry or equivalent error tracking for unhandled exceptions.
- Emit Prometheus or application metrics for request volume, latency, and failed requests.
- Set alert thresholds for infrastructure health, failed login spikes, and expensive provider changes.

## 4. Budget controls and chargeback

- Require cost tagging for every resource and workload.
- Track owner and business unit on every scenario or estimate.
- Alert when the monthly cost forecast exceeds threshold or deviates sharply from historical norms.
- Build cost allocation reports by team, project, and cloud provider.

## Recommended rollout order

1. Authentication + RBAC
2. PostgreSQL-based persistence
3. Secret management and HTTPS
4. Monitoring and alerting
5. Budget alerts and chargeback
