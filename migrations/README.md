# Database migration approach

This project uses a simple migration-based pattern that is compatible with a production-grade PostgreSQL or MySQL deployment.

## Recommended production database

- PostgreSQL for enterprise workloads, transactional safety, and strong tooling support
- MySQL as a valid alternative where the deployment environment already standardizes on it

## Migration workflow

1. Add a new migration file under `migrations/versions/`.
2. Name the file in the format `YYYYMMDDHHMMSS_<description>.sql`.
3. Keep each migration idempotent and explicit.
4. Store forward changes in SQL or ORM migration scripts.
5. Run the migration from a CI/CD environment with an admin-grade database user.

## Example migration commands

```bash
# PostgreSQL
psql "$DATABASE_URL" -f migrations/versions/20260816000000_create_users_table.sql

# MySQL
mysql --defaults-extra-file=.my.cnf -f < migrations/versions/20260816000000_create_users_table.sql
```

## Operational guidance

- Use managed database services in production.
- Keep schema changes in source control.
- Test migrations in staging before production rollout.
- Use separate read/write credentials and least-privilege access.
