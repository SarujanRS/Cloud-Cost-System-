"""
auth_store.py
-------------
Real (if minimal) user accounts: username + hashed password, backed by SQLite
via the stdlib sqlite3 module and Werkzeug's password hashing (already a
Flask dependency, no new package needed).

Scope note: this is a single self-hosted Flask process. Password hashes are
salted and never stored or transmitted in plaintext. The session cookie is
HttpOnly/SameSite=Lax with a signing key persisted to pricing/secret.key
(see backend/app.py), but the session cookie and login POST body only get
transport encryption once you run over HTTPS - either FLASK_HTTPS=1 for a
local self-signed test, or a real reverse proxy + FLASK_COOKIE_SECURE=1 in
production. Fine for local/dev use as shipped; turn on HTTPS before a
public deployment.
"""

import sqlite3
import datetime as dt
from pathlib import Path

from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = Path(__file__).with_name("auth.db")
VALID_ROLES = ("Owner", "Editor", "Viewer")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'Owner',
                created_at TEXT NOT NULL
            )
        """)


def _now():
    return dt.datetime.utcnow().isoformat() + "Z"


def _public(row):
    if row is None:
        return None
    return {"id": row["id"], "username": row["username"], "email": row["email"],
            "display_name": row["display_name"], "role": row["role"], "created_at": row["created_at"]}


def _normalize_username(username):
    # Usernames are matched case-insensitively (stored lowercased) so "Bob" and "bob"
    # are the same account - browsers inconsistently auto-capitalize the first letter
    # of text inputs, which otherwise turns into a silent "invalid password" for users
    # who typed the same name with different casing at register vs. login time.
    return (username or "").strip().lower()


def create_user(username, password, display_name=None, email=None):
    display_name = (display_name or "").strip() or (username or "").strip()
    username = _normalize_username(username)
    if len(username) < 3:
        return None, "Username must be at least 3 characters"
    if not password or len(password) < 6:
        return None, "Password must be at least 6 characters"
    with _connect() as conn:
        if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            return None, "That username is already taken"
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, display_name, role, created_at) "
            "VALUES (?, ?, ?, ?, 'Owner', ?)",
            (username, (email or "").strip() or None, generate_password_hash(password),
             display_name, _now()),
        )
        return cur.lastrowid, None


def verify_user(username, password):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (_normalize_username(username),)).fetchone()
    if not row or not check_password_hash(row["password_hash"], password or ""):
        return None
    return _public(row)


def get_user(user_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _public(row)


def update_profile(user_id, display_name=None, role=None):
    fields, params = [], []
    if display_name is not None and display_name.strip():
        fields.append("display_name = ?")
        params.append(display_name.strip()[:80])
    if role is not None and role in VALID_ROLES:
        fields.append("role = ?")
        params.append(role)
    if not fields:
        return get_user(user_id)
    params.append(user_id)
    with _connect() as conn:
        conn.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", params)
    return get_user(user_id)


def change_password(user_id, current_password, new_password):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row or not check_password_hash(row["password_hash"], current_password or ""):
            return False, "Current password is incorrect"
        if not new_password or len(new_password) < 6:
            return False, "New password must be at least 6 characters"
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                     (generate_password_hash(new_password), user_id))
    return True, None


init_db()
