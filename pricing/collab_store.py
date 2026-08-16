"""
collab_store.py
----------------
Lightweight "collaboration" persistence for shared scenarios and their
comment threads, backed by SQLite (stdlib only, no new dependency).

There is no authentication here. "Role" is a label the sharer/commenter
chooses client-side (Owner / Editor / Viewer) and is stored purely for
display and for the frontend's own read-only gating - it is NOT an access
control boundary. Anyone with a share link can view or comment. That's an
intentional scope decision for a lightweight collaboration feature, not an
oversight; real access control would need real accounts.
"""

import json
import secrets
import sqlite3
import datetime as dt
from pathlib import Path

DB_PATH = Path(__file__).with_name("collab.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_scenarios (
                id TEXT PRIMARY KEY,
                scenario_json TEXT NOT NULL,
                owner_name TEXT,
                role TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scenario_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scenario_id TEXT NOT NULL,
                author TEXT,
                role TEXT,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (scenario_id) REFERENCES shared_scenarios(id)
            )
        """)


def _now():
    return dt.datetime.utcnow().isoformat() + "Z"


def create_share(scenario, owner_name, role):
    share_id = secrets.token_urlsafe(6)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO shared_scenarios (id, scenario_json, owner_name, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (share_id, json.dumps(scenario), (owner_name or "Anonymous")[:80], (role or "Owner")[:20], _now()),
        )
    return share_id


def get_share(share_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM shared_scenarios WHERE id = ?", (share_id,)).fetchone()
        if row is None:
            return None
        comments = conn.execute(
            "SELECT author, role, text, created_at FROM scenario_comments WHERE scenario_id = ? ORDER BY id ASC",
            (share_id,),
        ).fetchall()
    return {
        "id": row["id"], "scenario": json.loads(row["scenario_json"]),
        "owner_name": row["owner_name"], "role": row["role"], "created_at": row["created_at"],
        "comments": [dict(c) for c in comments],
    }


def add_comment(share_id, author, role, text):
    with _connect() as conn:
        exists = conn.execute("SELECT 1 FROM shared_scenarios WHERE id = ?", (share_id,)).fetchone()
        if exists is None:
            return None
        conn.execute(
            "INSERT INTO scenario_comments (scenario_id, author, role, text, created_at) VALUES (?, ?, ?, ?, ?)",
            (share_id, (author or "Anonymous")[:80], (role or "Viewer")[:20], text[:2000], _now()),
        )
        comments = conn.execute(
            "SELECT author, role, text, created_at FROM scenario_comments WHERE scenario_id = ? ORDER BY id ASC",
            (share_id,),
        ).fetchall()
    return [dict(c) for c in comments]


init_db()
