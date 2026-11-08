"""Tests for search_events (PRD FR-3). Run: python -m pytest tests/test_search.py -v"""
import json

import pytest

from app import db
from app.agent import run_agent
from app.tools import TOOL_SCHEMAS, search_events

from tests.test_agent import Scripted, call, text


def make_image(conn, sha, created_at, confidences, label="car"):
    iid = db.insert_image(conn, sha, 640, 480)
    db.insert_detections(conn, iid,
                         [{"label": label, "confidence": c} for c in confidences], "v1")
    conn.execute("UPDATE images SET created_at = ? WHERE id = ?", (created_at, iid))
    conn.commit()
    return iid


@pytest.fixture
def week(conn):
    """Three uploads on different dates, plus one image with only a low-confidence car."""
    ids = {}
    ids["oct20"] = make_image(conn, "s-oct20", "2026-10-20T09:00:00+00:00", [0.9, 0.9, 0.9])
    ids["nov2"] = make_image(conn, "s-nov2", "2026-11-02T10:00:00+00:00", [0.9, 0.8])
    ids["nov5"] = make_image(conn, "s-nov5", "2026-11-05T12:00:00+00:00", [0.9, 0.1])
    ids["nov6"] = make_image(conn, "s-nov6", "2026-11-06T08:00:00+00:00", [0.1])
    return ids


def test_date_range_is_inclusive_and_newest_first(conn, week):
    out = search_events(conn, None, "car", "2026-11-01", "2026-11-05")
    assert [r["image_id"] for r in out] == [week["nov5"], week["nov2"]]


def test_counts_respect_default_threshold(conn, week):
    out = {r["image_id"]: r["count"]
           for r in search_events(conn, None, "car", "2026-11-01", "2026-11-08")}
    assert out[week["nov2"]] == 2
    assert out[week["nov5"]] == 1  # the 0.1 car is below 0.25
    assert week["nov6"] not in out  # only a low-confidence car


def test_lower_threshold_includes_low_confidence(conn, week):
    out = {r["image_id"]: r["count"]
           for r in search_events(conn, None, "car", "2026-11-01", "2026-11-08",
                                  min_confidence=0.1)}
    assert out[week["nov5"]] == 2
    assert out[week["nov6"]] == 1


def test_limit_applies_after_ordering(conn, week):
    out = search_events(conn, None, "car", "2026-10-01", "2026-11-08", limit=1)
    assert [r["image_id"] for r in out] == [week["nov5"]]  # newest first, then stop


def test_label_is_case_insensitive(conn, week):
    assert len(search_events(conn, None, "CAR", "2026-11-01", "2026-11-08")) == 2


def test_absent_label_returns_empty_list(conn, week):
    assert search_events(conn, None, "person", "2026-10-01", "2026-11-08") == []


def test_result_shape(conn, week):
    row = search_events(conn, None, "car", "2026-11-01", "2026-11-08")[0]
    assert set(row) == {"image_id", "created_at", "count"}


def test_image_id_argument_is_ignored_for_cross_image_search(conn, week):
    a = search_events(conn, "no-such-image", "car", "2026-11-01", "2026-11-08")
    b = search_events(conn, week["nov2"], "car", "2026-11-01", "2026-11-08")
    assert a == b


@pytest.mark.parametrize("since, until", [
    ("2026-11-06", "2026-11-01"),   # since after until
    ("2026/11/01", "2026-11-08"),   # wrong separator
    ("20261101", "2026-11-08"),     # compact form
    ("2026-13-01", "2026-11-08"),   # invalid month
    ("2026-02-30", "2026-11-08"),   # invalid day
    (None, "2026-11-08"),
    ("2026-11-01", 20261108),
])
def test_bad_dates_rejected(conn, week, since, until):
    with pytest.raises(ValueError):
        search_events(conn, None, "car", since, until)


@pytest.mark.parametrize("bad", [0, 51, True, "5", 2.5])
def test_limit_bounds_rejected(conn, week, bad):
    with pytest.raises(ValueError):
        search_events(conn, None, "car", "2026-11-01", "2026-11-08", limit=bad)


@pytest.mark.parametrize("bad", ["", "   ", "x" * 41, None])
def test_label_rejected(conn, week, bad):
    with pytest.raises(ValueError):
        search_events(conn, None, bad, "2026-11-01", "2026-11-08")


def test_hostile_label_is_bound_not_executed(conn, week):
    hostile = "car' OR '1'='1"
    assert search_events(conn, None, hostile, "2026-11-01", "2026-11-08") == []
    assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 4


def test_schema_exposes_no_image_fields_and_requires_dates():
    schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "search_events")
    props = schema["function"]["parameters"]["properties"]
    assert "image_id" not in props
    assert set(schema["function"]["parameters"]["required"]) == {"label", "since", "until"}


def test_agent_runs_search_events_and_records_it(conn, week):
    llm = Scripted([
        call("search_events",
             {"label": "car", "since": "2026-11-01", "until": "2026-11-08"}, "c1"),
        text("Two images had cars this week."),
    ])
    res = run_agent(conn, "Which uploads had cars this week?", week["nov2"], llm)

    assert res["status"] == "ok"
    calls = db.get_tool_calls(conn, res["run_id"])
    assert calls[0]["tool_name"] == "search_events"
    assert calls[0]["ok"] == 1
    assert json.loads(calls[0]["result_summary"])[0]["image_id"] == week["nov5"]


def test_agent_cannot_pass_image_id_to_search_events(conn, week):
    llm = Scripted([
        call("search_events",
             {"label": "car", "since": "2026-11-01", "until": "2026-11-08",
              "image_id": "other"}, "c1"),
        text("done"),
    ])
    res = run_agent(conn, "q", week["nov2"], llm)
    calls = db.get_tool_calls(conn, res["run_id"])
    assert calls[0]["ok"] == 0
    assert "image_id" in calls[0]["error"]
