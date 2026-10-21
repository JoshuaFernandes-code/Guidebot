"""
emails_db.py — SQLite-backed registry of allowed dashboard login emails.

Schema
------
allowed_emails(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    email       TEXT NOT NULL UNIQUE,   -- lowercased, trimmed email address
    added_at    TEXT NOT NULL,          -- ISO-8601 UTC timestamp
    added_by    TEXT,                   -- optional: who added this email
    password_hash        TEXT,          -- per-user PBKDF2-HMAC-SHA256 hash (NULL = not set)
    password_updated_at  TEXT           -- ISO-8601 UTC of last password change (NULL = never)
)

settings(
    setting_key   TEXT PRIMARY KEY,     -- legacy: "password_hash", "password_updated_at"
    setting_value TEXT NOT NULL
)

dashboard_sessions(
    token       TEXT PRIMARY KEY,
    email       TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    pw_ts       TEXT NOT NULL
)

Each allowed email has its OWN password, stored ONLY as a PBKDF2-HMAC-SHA256
hash (never plaintext) and manageable only through update_email.py.
Changing a user's password bumps their password_updated_at, which
invalidates only that user's existing sessions.

Legacy migration: very old DBs stored ONE shared password in settings.
On first connect it is copied to every existing user that has no password
yet, then removed from settings.

All queries use parameterized statements — SQL injection is not possible.
"""

import hashlib
import hmac
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone

MIN_PASSWORD_LENGTH = 8
_PBKDF2_ITERATIONS = 600_000
_LEGACY_PASSWORD_KEY = "password_hash"
_LEGACY_PASSWORD_TS_KEY = "password_updated_at"

DB_PATH = os.path.join("data", "allowed_emails.db")

# Module-level lock so multiple threads share one connection safely
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """Return (and lazily create) the shared SQLite connection."""
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _create_tables(_conn)
    return _conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS allowed_emails (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            email       TEXT NOT NULL UNIQUE,
            added_at    TEXT NOT NULL,
            added_by    TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            setting_key   TEXT PRIMARY KEY,
            setting_value TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dashboard_sessions (
            token       TEXT PRIMARY KEY,
            email       TEXT NOT NULL,
            expires_at  TEXT NOT NULL,
            pw_ts       TEXT NOT NULL
        )
    """)
    # Per-user password columns (added after the shared-password era).
    for _ddl in (
        "ALTER TABLE allowed_emails ADD COLUMN password_hash TEXT",
        "ALTER TABLE allowed_emails ADD COLUMN password_updated_at TEXT",
    ):
        try:
            conn.execute(_ddl)
        except sqlite3.OperationalError:
            pass  # column already exists
    _migrate_legacy_shared_password(conn)
    conn.commit()


def _migrate_legacy_shared_password(conn: sqlite3.Connection) -> None:
    """One-time migration: copy the old shared password onto every user.

    Very old DBs stored ONE global password_hash (+timestamp) in settings.
    Copy it to each existing user that has no password yet, then delete the
    legacy rows so the shared credential cannot linger.
    """
    row = conn.execute(
        "SELECT setting_value FROM settings WHERE setting_key = ?",
        (_LEGACY_PASSWORD_KEY,),
    ).fetchone()
    if row is None:
        return
    legacy_hash = row["setting_value"]
    ts_row = conn.execute(
        "SELECT setting_value FROM settings WHERE setting_key = ?",
        (_LEGACY_PASSWORD_TS_KEY,),
    ).fetchone()
    legacy_ts = ts_row["setting_value"] if ts_row else datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE allowed_emails SET password_hash = ?, password_updated_at = ? "
        "WHERE password_hash IS NULL",
        (legacy_hash, legacy_ts),
    )
    conn.execute(
        "DELETE FROM settings WHERE setting_key IN (?, ?)",
        (_LEGACY_PASSWORD_KEY, _LEGACY_PASSWORD_TS_KEY),
    )


def _normalize_email(email: str) -> str:
    """Lowercase and trim an email address."""
    return email.strip().lower()


def _is_valid_email(email: str) -> bool:
    """Basic email format validation."""
    email = email.strip()
    if not email or "@" not in email:
        return False
    local, _, domain = email.partition("@")
    if not local or not domain or "." not in domain:
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_email(email: str, added_by: str | None = None) -> dict:
    """Add an allowed email address.

    Returns the inserted row as a dict. Raises ValueError if the email is
    invalid or already exists.
    """
    email = _normalize_email(email)
    if not _is_valid_email(email):
        raise ValueError(f"Invalid email address: {email}")

    with _lock:
        conn = _get_conn()
        existing = conn.execute(
            "SELECT id FROM allowed_emails WHERE email = ?", (email,)
        ).fetchone()
        if existing is not None:
            raise ValueError(f"Email already registered: {email}")

        added_at = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT INTO allowed_emails (email, added_at, added_by) VALUES (?, ?, ?)",
            (email, added_at, added_by),
        )
        conn.commit()
        row_id = cur.lastrowid

    return {
        "id": row_id,
        "email": email,
        "added_at": added_at,
        "added_by": added_by,
    }


def list_emails() -> list[dict]:
    """Return all allowed emails ordered by added_at descending."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, email, added_at, added_by FROM allowed_emails ORDER BY added_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_email(email: str) -> bool:
    """Delete an allowed email by address. Returns True if found and deleted."""
    email = _normalize_email(email)
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT id FROM allowed_emails WHERE email = ?", (email,)
        ).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM allowed_emails WHERE id = ?", (row["id"],))
        conn.commit()
    return True


def delete_email_by_id(email_id: int) -> bool:
    """Delete an allowed email by ID. Returns True if found and deleted."""
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "DELETE FROM allowed_emails WHERE id = ?", (email_id,)
        )
        conn.commit()
        return cur.rowcount > 0


def is_allowed(email: str) -> bool:
    """Return True if the given email is in the allowed list."""
    email = _normalize_email(email)
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT 1 FROM allowed_emails WHERE email = ?", (email,)
        ).fetchone()
    return row is not None


def get_email_by_id(email_id: int) -> dict | None:
    """Return an allowed email record by ID, or None if not found."""
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT id, email, added_at, added_by FROM allowed_emails WHERE id = ?",
            (email_id,),
        ).fetchone()
    return dict(r) if (r := row) else None


# ---------------------------------------------------------------------------
# Per-user dashboard passwords (managed exclusively via update_email.py)
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 + random 16-byte salt."""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def _check_hash(password: str, stored: str | None) -> bool:
    """Constant-time check of a candidate password against a stored hash."""
    if not password or not stored:
        return False
    try:
        algo, iters_s, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iters_s)
        salt = bytes.fromhex(salt_hex)
    except (ValueError, AttributeError):
        return False
    # PBKDF2 runs outside any lock so concurrent requests are not serialized.
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk.hex(), hash_hex)


def set_password(email: str, password: str) -> None:
    """Hash and store the password for ONE allowed email.

    Raises ValueError if the email is not registered or the password is too
    short. Rotating the password bumps that user's password_updated_at,
    which invalidates only that user's existing sessions.
    """
    email = _normalize_email(email)
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")

    stored = _hash_password(password)
    updated_at = datetime.now(timezone.utc).isoformat()

    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "UPDATE allowed_emails SET password_hash = ?, password_updated_at = ? "
            "WHERE email = ?",
            (stored, updated_at, email),
        )
        conn.commit()
    if cur.rowcount == 0:
        raise ValueError(f"Email not registered: {email}")


def has_password(email: str | None = None) -> bool:
    """Return True if a password is configured.

    With an email: checks that specific user. Without: True if ANY
    registered user has a password set.
    """
    with _lock:
        conn = _get_conn()
        if email is not None:
            row = conn.execute(
                "SELECT 1 FROM allowed_emails WHERE email = ? AND password_hash IS NOT NULL",
                (_normalize_email(email),),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT 1 FROM allowed_emails WHERE password_hash IS NOT NULL LIMIT 1"
            ).fetchone()
    return row is not None


def verify_password(email: str, password: str) -> bool:
    """Constant-time check of a candidate password for ONE allowed email."""
    email = _normalize_email(email)
    if not password:
        return False
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT password_hash FROM allowed_emails WHERE email = ?",
            (email,),
        ).fetchone()
    if row is None:
        return False
    return _check_hash(password, row["password_hash"])


def get_password_updated_at(email: str) -> str | None:
    """Return the ISO timestamp of a user's last password change, or None."""
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT password_updated_at FROM allowed_emails WHERE email = ?",
            (_normalize_email(email),),
        ).fetchone()
    return row["password_updated_at"] if row and row["password_updated_at"] else None


def create_dashboard_session(
    token: str, email: str, expires_at: datetime, pw_ts: str
) -> None:
    """Persist an opaque dashboard session until expiry or explicit logout."""
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO dashboard_sessions (token, email, expires_at, pw_ts) "
            "VALUES (?, ?, ?, ?)",
            (token, _normalize_email(email), expires_at.isoformat(), pw_ts),
        )
        conn.execute("DELETE FROM dashboard_sessions WHERE expires_at <= ?", (now,))
        conn.commit()


def get_dashboard_session(token: str) -> dict | None:
    """Return the persisted session record for an opaque token, if present."""
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT email, expires_at, pw_ts FROM dashboard_sessions WHERE token = ?",
            (token,),
        ).fetchone()
    return dict(row) if row else None


def delete_dashboard_session(token: str) -> None:
    """Revoke a dashboard session token."""
    with _lock:
        conn = _get_conn()
        conn.execute("DELETE FROM dashboard_sessions WHERE token = ?", (token,))
        conn.commit()


def users_without_password() -> list:
    """Return allowed emails that have no password set yet."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, email, added_at, added_by FROM allowed_emails "
            "WHERE password_hash IS NULL ORDER BY email"
        ).fetchall()
    return [dict(r) for r in rows]