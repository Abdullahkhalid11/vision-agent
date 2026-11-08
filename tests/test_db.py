"""Tests for app/db.py. Run from repo root: python -m pytest tests/test_db.py -v"""
import sqlite3
from contextlib import closing

import pytest

from app import db


@pytest.fixture
def conn(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    c = db.connect(path)
    yield c
    c.close()


def test_schema_creates_four_tables_and_wal(tmp_path):
    path = tmp_path / "s.db"
    db.init_db(path)
    with closing(db.connect(path)) as c:
        names = {r["name"] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        mode = c.execute("PRAGMA journal_mode").fetchone()[0]
    assert {"images", "detections", "agent_runs", "tool_calls"} <= names
    assert mode == "wal"


def test_init_db_is_idempotent(tmp_path):
    path = tmp_path / "i.db"
    db.init_db(path)
    db.init_db(path)  # must not raise or drop data
    with closing(db.connect(path)) as c:
        c.execute("SELECT COUNT(*) FROM images").fetchone()


def test_insert_image_dedupes_on_sha256(conn):
    a = db.insert_image(conn, "abc", 640, 480)
    b = db.insert_image(conn, "abc", 640, 480)
    assert a == b
    count = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    assert count == 1


def test_get_image_roundtrip(conn):
    image_id = db.insert_image(conn, "sha1", 100, 50)
    row = db.get_image(conn, image_id)
    assert row["sha256"] == "sha1"
    assert (row["width"], row["height"]) == (100, 50)
    assert db.get_image_by_sha(conn, "sha1")["id"] == image_id
    assert db.get_image(conn, "missing") is None


def test_detections_roundtrip_and_threshold_filter(conn):
    image_id = db.insert_image(conn, "sha2", 640, 640)
    dets = [
        {"label": "car", "confidence": 0.9, "x1": 1, "y1": 2, "x2": 3, "y2": 4},
        {"label": "car", "confidence": 0.3, "x1": 5, "y1": 6, "x2": 7, "y2": 8},
        {"label": "person", "confidence": 0.6, "x1": 0, "y1": 0, "x2": 9, "y2": 9},
    ]
    assert db.insert_detections(conn, image_id, dets, "yolov8n-onnx") == 3
    assert db.has_detections(conn, image_id) is True
    assert len(db.get_detections(conn, image_id)) == 3
    high = db.get_detections(conn, image_id, min_confidence=0.5)
    assert [d["label"] for d in high] == ["car", "person"]


def test_has_detections_false_when_none(conn):
    image_id = db.insert_image(conn, "sha3", 10, 10)
    assert db.has_detections(conn, image_id) is False


def test_foreign_key_enforced_on_detections(conn):
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_detections(
            conn, "no-such-image",
            [{"label": "car", "confidence": 0.9}], "v1",
        )


def test_run_lifecycle_and_tool_calls(conn):
    image_id = db.insert_image(conn, "sha4", 640, 480)
    run_id = db.start_run(conn, "How many cars?", image_id)
    assert db.get_run(conn, run_id)["status"] == "running"

    db.insert_tool_call(conn, run_id, 1, "detect_objects",
                        '{"image_id": "x"}', "18 detections", 71, True)
    db.insert_tool_call(conn, run_id, 2, "count_labels",
                        '{"label": "car"}', '{"count": 17}', 2, True)
    db.finish_run(conn, run_id, "ok", "There are 17 cars.", steps=2,
                  total_latency_ms=1500, input_tokens=900, output_tokens=40)

    run = db.get_run(conn, run_id)
    assert run["status"] == "ok"
    assert run["answer"] == "There are 17 cars."
    assert run["steps"] == 2
    calls = db.get_tool_calls(conn, run_id)
    assert [c["tool_name"] for c in calls] == ["detect_objects", "count_labels"]
    assert [c["ok"] for c in calls] == [1, 1]


def test_tool_call_result_summary_truncated_to_500(conn):
    run_id = db.start_run(conn, "q", None)
    db.insert_tool_call(conn, run_id, 1, "detect_objects", "{}", "x" * 2000, 1, True)
    stored = db.get_tool_calls(conn, run_id)[0]["result_summary"]
    assert len(stored) == 500


def test_finish_run_rejects_invalid_status(conn):
    run_id = db.start_run(conn, "q", None)
    with pytest.raises(ValueError):
        db.finish_run(conn, run_id, "running", None, 0, 0)
    with pytest.raises(ValueError):
        db.finish_run(conn, run_id, "deleted_everything", None, 0, 0)


def test_finish_run_unknown_id_raises(conn):
    with pytest.raises(KeyError):
        db.finish_run(conn, "nope", "ok", "a", 0, 0)


def test_tool_call_requires_existing_run(conn):
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_tool_call(conn, "ghost-run", 1, "detect_objects", "{}", None, 1, True)


def test_sql_injection_string_stored_literally_and_table_survives(conn):
    hostile = "car'); DROP TABLE detections; --"
    image_id = db.insert_image(conn, "sha5", 10, 10)
    db.insert_detections(conn, image_id,
                         [{"label": hostile, "confidence": 0.9}], "v1")
    got = db.get_detections(conn, image_id)
    assert got[0]["label"] == hostile
    assert db.has_detections(conn, image_id)
