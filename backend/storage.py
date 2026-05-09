"""SQLite-backed persistence for documents, chats, and messages.

A single sqlite file (uploaded_files/pdf_agent.db) stores:
  - documents: one row per uploaded PDF (keyed by file_hash, with doc_id)
  - chats:     one row per conversation, tied to a document
  - messages:  one row per turn (user or assistant) within a chat

The retriever objects themselves are not persisted here — those are rebuilt at
startup from the on-disk Chroma persist_dir referenced by each document row.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


DB_PATH = Path("uploaded_files") / "pdf_agent.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    file_hash     TEXT NOT NULL UNIQUE,
    filename      TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    persist_dir   TEXT NOT NULL,
    chunk_count   INTEGER,
    size_bytes    INTEGER,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chats (
    chat_id       TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    title         TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chats_doc      ON chats(doc_id);
CREATE INDEX IF NOT EXISTS idx_chats_updated  ON chats(updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    message_id    TEXT PRIMARY KEY,
    chat_id       TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
    role          TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content       TEXT NOT NULL,
    created_at    REAL NOT NULL,
    metrics_json  TEXT,
    sources_json  TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at);
"""


_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    _conn = conn
    return conn


def init() -> None:
    """Open the connection and ensure schema. Safe to call multiple times."""
    _connect()


# ---------- documents ----------

def upsert_document(
    *,
    doc_id: str,
    file_hash: str,
    filename: str,
    file_path: str,
    persist_dir: str,
    chunk_count: Optional[int],
    size_bytes: Optional[int],
) -> None:
    with _lock:
        _connect().execute(
            """
            INSERT INTO documents (doc_id, file_hash, filename, file_path, persist_dir,
                                   chunk_count, size_bytes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_hash) DO UPDATE SET
                filename=excluded.filename,
                file_path=excluded.file_path,
                persist_dir=excluded.persist_dir,
                chunk_count=excluded.chunk_count,
                size_bytes=excluded.size_bytes
            """,
            (doc_id, file_hash, filename, file_path, persist_dir,
             chunk_count, size_bytes, time.time()),
        )


def get_document_by_hash(file_hash: str) -> Optional[dict]:
    row = _connect().execute(
        "SELECT * FROM documents WHERE file_hash=?", (file_hash,)
    ).fetchone()
    return dict(row) if row else None


def get_document(doc_id: str) -> Optional[dict]:
    row = _connect().execute(
        "SELECT * FROM documents WHERE doc_id=?", (doc_id,)
    ).fetchone()
    return dict(row) if row else None


def list_documents() -> list[dict]:
    rows = _connect().execute(
        "SELECT * FROM documents ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


# ---------- chats ----------

def create_chat(doc_id: str, title: Optional[str] = None) -> dict:
    chat_id = uuid4().hex
    now = time.time()
    with _lock:
        _connect().execute(
            "INSERT INTO chats (chat_id, doc_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (chat_id, doc_id, title, now, now),
        )
    return {"chat_id": chat_id, "doc_id": doc_id, "title": title,
            "created_at": now, "updated_at": now}


def get_chat(chat_id: str) -> Optional[dict]:
    row = _connect().execute(
        "SELECT * FROM chats WHERE chat_id=?", (chat_id,)
    ).fetchone()
    return dict(row) if row else None


def list_chats() -> list[dict]:
    """Return all chats with their associated document filename, newest first."""
    rows = _connect().execute(
        """
        SELECT c.chat_id, c.doc_id, c.title, c.created_at, c.updated_at,
               d.filename, d.chunk_count, d.size_bytes,
               (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.chat_id
                  AND m.role IN ('user','assistant')) AS message_count
        FROM chats c
        JOIN documents d ON d.doc_id = c.doc_id
        ORDER BY c.updated_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def rename_chat(chat_id: str, title: str) -> bool:
    with _lock:
        cur = _connect().execute(
            "UPDATE chats SET title=?, updated_at=? WHERE chat_id=?",
            (title, time.time(), chat_id),
        )
        return cur.rowcount > 0


def touch_chat(chat_id: str) -> None:
    with _lock:
        _connect().execute(
            "UPDATE chats SET updated_at=? WHERE chat_id=?",
            (time.time(), chat_id),
        )


def delete_chat(chat_id: str) -> bool:
    with _lock:
        cur = _connect().execute("DELETE FROM chats WHERE chat_id=?", (chat_id,))
        return cur.rowcount > 0


# ---------- messages ----------

def add_message(
    *,
    chat_id: str,
    role: str,
    content: str,
    metrics: Optional[dict] = None,
    sources: Optional[list] = None,
) -> dict:
    message_id = uuid4().hex
    now = time.time()
    with _lock:
        _connect().execute(
            """
            INSERT INTO messages (message_id, chat_id, role, content, created_at,
                                  metrics_json, sources_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id, chat_id, role, content, now,
                json.dumps(metrics) if metrics else None,
                json.dumps(sources) if sources else None,
            ),
        )
        _connect().execute(
            "UPDATE chats SET updated_at=? WHERE chat_id=?", (now, chat_id),
        )
    return {"message_id": message_id, "created_at": now}


def list_messages(chat_id: str) -> list[dict]:
    rows = _connect().execute(
        "SELECT message_id, role, content, created_at, metrics_json, sources_json "
        "FROM messages WHERE chat_id=? ORDER BY created_at ASC, message_id ASC",
        (chat_id,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["metrics"] = json.loads(d.pop("metrics_json")) if d.get("metrics_json") else None
        d["sources"] = json.loads(d.pop("sources_json")) if d.get("sources_json") else None
        out.append(d)
    return out


def history_for_prompt(chat_id: str, max_turns: int = 5) -> list[dict[str, Any]]:
    """Return the last `max_turns` user/assistant pairs for prompt context."""
    rows = _connect().execute(
        "SELECT role, content FROM messages WHERE chat_id=? AND role IN ('user','assistant') "
        "ORDER BY created_at ASC, message_id ASC",
        (chat_id,),
    ).fetchall()
    pairs: list[dict[str, str]] = []
    pending_user: Optional[str] = None
    for r in rows:
        if r["role"] == "user":
            pending_user = r["content"]
        elif r["role"] == "assistant" and pending_user is not None:
            pairs.append({"user": pending_user, "assistant": r["content"]})
            pending_user = None
    return pairs[-max_turns:]
