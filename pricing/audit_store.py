"""
audit_store.py
---------------
Immutable-by-convention audit trail: who did what, when. Backed by SQLite
(stdlib only, no new dependency), same pattern as auth_store.py / collab_store.py.

"Immutable-by-convention" - there's no UPDATE or DELETE anywhere in this module,
only INSERT and SELECT, so nothing in this app's own code path can alter or
remove a logged entry. That's enough for a local single-process dev tool; a
production deployment wanting a true tamper-evident trail would want this
written to an append-only/WORM store instead of a local SQLite file an
operator could still edit by hand.
"""

import csv
import io
import sqlite3
import datetime as dt
from pathlib import Path

DB_PATH = Path(__file__).with_name("audit.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                created_at TEXT NOT NULL
            )
        """)


def _now():
    return dt.datetime.utcnow().isoformat() + "Z"


def log_event(user_id, username, action, details=""):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO audit_log (user_id, username, action, details, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, (username or "anonymous")[:80], action[:60], (details or "")[:500], _now()),
        )


def list_events(limit=200):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, user_id, username, action, details, created_at FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def export_csv(limit=5000):
    rows = list_events(limit=limit)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "created_at", "user_id", "username", "action", "details"])
    for r in rows:
        writer.writerow([r["id"], r["created_at"], r["user_id"], r["username"], r["action"], r["details"]])
    return buf.getvalue()


init_db()
