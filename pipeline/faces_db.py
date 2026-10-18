"""
faces_db.py — SQLite-backed registry for known face images.

Schema
------
faces(
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,        -- human-readable label shown during recognition
    filename     TEXT NOT NULL UNIQUE, -- basename only, e.g. "nathan_1.png"
    added_at     TEXT NOT NULL         -- ISO-8601 UTC timestamp
)

Name → file convention
-----------------------
Files are stored as  <slug>_<N>.ext  where:
  slug  = display_name lowercased, spaces/special-chars replaced with '_'
  N     = 1-based counter to handle duplicates for the same person

The DB is the source of truth for display_name.  The face recogniser uses
display_name for the label it speaks/shows.  If a record is deleted from
the DB but the file remains, the recogniser falls back to the stem of the
filename as the display name (backward-compatible with legacy plain files).
"""

import os
import re
import sqlite3
import threading
from datetime import datetime, timezone


KNOWN_FACES_DIR = os.path.join("data", "known_faces")
DB_PATH = os.path.join(KNOWN_FACES_DIR, "faces.db")

_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Module-level lock so multiple threads share one connection safely
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """Return (and lazily create) the shared SQLite connection."""
    global _conn
    if _conn is None:
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _create_tables(_conn)
    return _conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            display_name TEXT    NOT NULL,
            filename     TEXT    NOT NULL UNIQUE,
            added_at     TEXT    NOT NULL
        )
    """)
    conn.commit()


def _slug(name: str) -> str:
    """Convert a display name to a safe filename slug.

    'Nathan Pereira' → 'nathan_pereira'
    'María José'     → 'maria_jose'
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name or "unknown"


def _next_filename(slug: str, ext: str) -> str:
    """Return the next available filename for *slug* inside KNOWN_FACES_DIR.

    Always uses a numeric suffix: slug_1.ext, slug_2.ext, …
    Checks both the filesystem AND the DB to avoid race-condition gaps.
    """
    conn = _get_conn()
    n = 1
    while True:
        candidate = f"{slug}_{n}{ext}"
        path_taken = os.path.exists(os.path.join(KNOWN_FACES_DIR, candidate))
        db_taken   = conn.execute(
            "SELECT 1 FROM faces WHERE filename = ?", (candidate,)
        ).fetchone() is not None
        if not path_taken and not db_taken:
            return candidate
        n += 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_face(display_name: str, image_bytes: bytes, ext: str = ".png") -> dict:
    """Save *image_bytes* to disk and register the face in the DB.

    Parameters
    ----------
    display_name : str   Human-readable label, e.g. "Nathan Pereira"
    image_bytes  : bytes Raw image file bytes (JPEG / PNG / WebP)
    ext          : str   File extension including the dot, default ".png"

    Returns
    -------
    dict  {id, display_name, filename, added_at, path}
    """
    ext = ext.lower()
    if ext not in _ALLOWED_EXTS:
        raise ValueError(f"Unsupported image extension: {ext!r}")

    if not display_name or not display_name.strip():
        raise ValueError("display_name must not be empty")

    slug = _slug(display_name)

    with _lock:
        filename = _next_filename(slug, ext)
        path     = os.path.join(KNOWN_FACES_DIR, filename)
        added_at = datetime.now(timezone.utc).isoformat()

        # Write file first; if DB insert fails we remove the file
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(image_bytes)

        try:
            conn = _get_conn()
            cur  = conn.execute(
                "INSERT INTO faces (display_name, filename, added_at) VALUES (?, ?, ?)",
                (display_name.strip(), filename, added_at),
            )
            conn.commit()
            row_id = cur.lastrowid
        except Exception:
            # Roll back the file write
            if os.path.exists(path):
                os.remove(path)
            raise

    return {
        "id":           row_id,
        "display_name": display_name.strip(),
        "filename":     filename,
        "added_at":     added_at,
        "path":         path,
    }


def list_faces() -> list[dict]:
    """Return all registered faces ordered by added_at descending."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, display_name, filename, added_at FROM faces ORDER BY added_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_face(face_id: int, delete_file: bool = True) -> bool:
    """Delete a face record from the DB and optionally remove the image file.

    Returns True if a record was found and deleted, False if not found.
    """
    with _lock:
        conn = _get_conn()
        row  = conn.execute(
            "SELECT filename FROM faces WHERE id = ?", (face_id,)
        ).fetchone()
        if row is None:
            return False

        filename = row["filename"]
        conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        conn.commit()

        if delete_file:
            path = os.path.join(KNOWN_FACES_DIR, filename)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass  # Log but don't crash

    return True


def update_face_name(face_id: int, new_display_name: str) -> bool:
    """Update the display_name of a face record.

    The filename on disk is NOT changed; only the human-readable display
    label is updated in the DB. The recogniser reads display_name via
    get_name_map(), so hot-reload picks up the change immediately.

    Returns True if a record was found and updated, False if not found.
    """
    if not new_display_name or not new_display_name.strip():
        raise ValueError("display_name must not be empty")

    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT id FROM faces WHERE id = ?", (face_id,)
        ).fetchone()
        if row is None:
            return False

        conn.execute(
            "UPDATE faces SET display_name = ? WHERE id = ?",
            (new_display_name.strip(), face_id),
        )
        conn.commit()

    return True


def get_name_map() -> dict[str, str]:
    """Return {filename: display_name} for all registered faces.

    Used by FaceRecognizer to map a loaded file to the correct label.
    """
    with _lock:
        conn = _get_conn()
        rows = conn.execute("SELECT filename, display_name FROM faces").fetchall()
    return {r["filename"]: r["display_name"] for r in rows}

