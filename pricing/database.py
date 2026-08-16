import os


def get_database_engine():
    engine = (os.getenv("DATABASE_ENGINE") or os.getenv("DB_ENGINE") or "sqlite").strip().lower()
    if engine in {"postgres", "postgresql"}:
        return "postgresql"
    if engine in {"mysql", "mariadb"}:
        return "mysql"
    if engine == "sqlite":
        return "sqlite"
    return "sqlite"


def get_database_url():
    return os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or os.getenv("MYSQL_URL") or ""


def get_database_status():
    engine = get_database_engine()
    database_url = get_database_url()
    return {
        "engine": engine,
        "configured": bool(database_url),
        "dialect": "postgresql" if engine == "postgresql" else "mysql" if engine == "mysql" else "sqlite",
        "migration_tool": "simple_sql_migrations",
        "recommended_for_production": engine in {"postgresql", "mysql"},
        "notes": "SQLite is acceptable for local development only; PostgreSQL or MySQL is recommended for production workloads.",
    }
