"""
SQLite persistence for Vision Agent (PRD V1, Section 6).

Rules:
- Parameterised queries only. No string formatting of values into SQL.
- No global connection. Callers open a connection with connect() and pass it in.
- Images are stored as hash and dimensions only, never as blobs.

Usage (from repo root):
    python -m app.db --db data/vision.db   # creates schema, prints table list
"""
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Optional

RESULT_SUMMARY_MAX = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
  id            TEXT PRIMARY KEY,
  sha256        TEXT NOT NULL UNIQUE,
  width         INTEGER NOT NULL,
  height        INTEGER NOT NULL,
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  image_id      TEXT NOT NULL REFERENCES images(id),
  label         TEXT NOT NULL,
  confidence    REAL NOT NULL,
  x1            REAL,
  y1            REAL,
  x2            REAL,
  y2            REAL,
  model_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_det_label ON detections(label, image_id);

CREATE TABLE IF NOT EXISTS agent_runs (
  id                 TEXT PRIMARY KEY,
  created_at         TEXT NOT NULL,
  image_id           TEXT REFERENCES images(id),
  question           TEXT NOT NULL,
  answer             TEXT,
  status             TEXT NOT NULL,
  steps              INTEGER NOT NULL DEFAULT 0,
  total_latency_ms   INTEGER,
  input_tokens       INTEGER,
  output_tokens      INTEGER,
  error              TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id          TEXT NOT NULL REFERENCES agent_runs(id),
  step            INTEGER NOT NULL,
  tool_name       TEXT NOT NULL,
  args_json       TEXT NOT NULL,
  result_summary  TEXT,
  latency_ms      INTEGER NOT NULL,
  ok              INTEGER NOT NULL,
  error           TEXT
);
"""

VALID_RUN_STATUS = {"ok", "max_steps", "llm_error", "tool_error", "rejected", "running"}


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with foreign keys on, WAL mode, and a 5 s busy timeout."""
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(db_path: str | Path) -> None:
    """Create all tables and indexes. Safe to call on every startup."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(db_path)) as conn, conn:
        conn.executescript(SCHEMA)


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    return dict(row) if row is not None else None


# ---------- images ----------

def insert_image(conn: sqlite3.Connection, sha256: str, width: int, height: int) -> str:
    """Insert an image or return the existing id for the same sha256 (dedupe / cache key)."""
    existing = conn.execute(
        "SELECT id FROM images WHERE sha256 = ?", (sha256,)
    ).fetchone()
    if existing is not None:
        return existing["id"]
    image_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO images (id, sha256, width, height, created_at) VALUES (?, ?, ?, ?, ?)",
        (image_id, sha256, width, height, utc_now_iso()),
    )
    conn.commit()
    return image_id


def get_image_by_sha(conn: sqlite3.Connection, sha256: str) -> Optional[dict[str, Any]]:
    return _row_to_dict(
        conn.execute("SELECT * FROM images WHERE sha256 = ?", (sha256,)).fetchone()
    )


def get_image(conn: sqlite3.Connection, image_id: str) -> Optional[dict[str, Any]]:
    return _row_to_dict(
        conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    )


# ---------- detections ----------

def insert_detections(
    conn: sqlite3.Connection,
    image_id: str,
    detections: Iterable[dict[str, Any]],
    model_version: str,
) -> int:
    """Store detections for one image. Each dict needs label, confidence, x1, y1, x2, y2."""
    rows = [
        (
            image_id,
            d["label"],
            float(d["confidence"]),
            d.get("x1"),
            d.get("y1"),
            d.get("x2"),
            d.get("y2"),
            model_version,
        )
        for d in detections
    ]
    conn.executemany(
        "INSERT INTO detections (image_id, label, confidence, x1, y1, x2, y2, model_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def has_detections(conn: sqlite3.Connection, image_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM detections WHERE image_id = ? LIMIT 1", (image_id,)
    ).fetchone()
    return row is not None


def get_detections(
    conn: sqlite3.Connection,
    image_id: str,
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT label, confidence, x1, y1, x2, y2 FROM detections "
        "WHERE image_id = ? AND confidence >= ? ORDER BY confidence DESC",
        (image_id, min_confidence),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------- agent runs and tool calls ----------

def start_run(
    conn: sqlite3.Connection,
    question: str,
    image_id: Optional[str],
) -> str:
    """Create an agent_runs row with status 'running'. Returns run_id."""
    run_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO agent_runs (id, created_at, image_id, question, status, steps) "
        "VALUES (?, ?, ?, ?, 'running', 0)",
        (run_id, utc_now_iso(), image_id, question),
    )
    conn.commit()
    return run_id


def finish_run(
    conn: sqlite3.Connection,
    run_id: str,
    status: str,
    answer: Optional[str],
    steps: int,
    total_latency_ms: int,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    if status not in VALID_RUN_STATUS or status == "running":
        raise ValueError(f"invalid final run status: {status!r}")
    cur = conn.execute(
        "UPDATE agent_runs SET status = ?, answer = ?, steps = ?, total_latency_ms = ?, "
        "input_tokens = ?, output_tokens = ?, error = ? WHERE id = ?",
        (status, answer, steps, total_latency_ms, input_tokens, output_tokens, error, run_id),
    )
    if cur.rowcount != 1:
        raise KeyError(f"no agent run with id {run_id!r}")
    conn.commit()


def insert_tool_call(
    conn: sqlite3.Connection,
    run_id: str,
    step: int,
    tool_name: str,
    args_json: str,
    result_summary: Optional[str],
    latency_ms: int,
    ok: bool,
    error: Optional[str] = None,
) -> int:
    summary = None if result_summary is None else result_summary[:RESULT_SUMMARY_MAX]
    cur = conn.execute(
        "INSERT INTO tool_calls (run_id, step, tool_name, args_json, result_summary, "
        "latency_ms, ok, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, step, tool_name, args_json, summary, latency_ms, 1 if ok else 0, error),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_run(conn: sqlite3.Connection, run_id: str) -> Optional[dict[str, Any]]:
    return _row_to_dict(
        conn.execute("SELECT * FROM agent_runs WHERE id = ?", (run_id,)).fetchone()
    )


def get_tool_calls(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT step, tool_name, args_json, result_summary, latency_ms, ok, error "
        "FROM tool_calls WHERE run_id = ? ORDER BY step, id",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Create Vision Agent SQLite schema.")
    parser.add_argument("--db", default="data/vision.db", help="path to SQLite file")
    args = parser.parse_args()
    init_db(args.db)
    with closing(connect(args.db)) as conn:
        tables = [
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
        ]
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    print(f"db: {args.db}")
    print(f"journal_mode: {mode}")
    print(f"tables: {', '.join(tables)}")


if __name__ == "__main__":
    _cli()
