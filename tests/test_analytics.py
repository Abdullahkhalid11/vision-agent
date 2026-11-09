"""Tests for app/analytics.py and db.get_run_rows. Run: python -m pytest tests/test_analytics.py -v"""
import pytest

from app import db
from app.analytics import summarise


def run(status, latency=None, tin=None, tout=None):
    return {"status": status, "total_latency_ms": latency,
            "input_tokens": tin, "output_tokens": tout}


def test_empty_database_is_safe():
    out = summarise([])
    assert out["runs"] == 0
    assert out["error_rate"] is None
    assert out["latency_ms"] == {"n": 0, "p50": None, "p95": None}
    assert out["estimated_cost_usd"] is None


def test_counts_statuses_and_error_rate():
    runs = [run("ok", 100, 10, 5), run("ok", 200, 10, 5),
            run("llm_error", 50), run("rejected", 0)]
    out = summarise(runs)
    assert out["runs"] == 4
    assert out["by_status"] == {"ok": 2, "llm_error": 1, "rejected": 1}
    assert out["error_rate"] == 0.25


def test_rejected_runs_excluded_from_latency():
    runs = [run("ok", 100), run("rejected", 0), run("ok", 300)]
    out = summarise(runs)
    assert out["latency_ms"]["n"] == 2
    assert out["latency_ms"]["p50"] == 100  # nearest rank: ceil(0.5 * 2) = 1


def test_percentiles():
    runs = [run("ok", ms) for ms in (100, 200, 300, 400, 500)]
    out = summarise(runs)
    assert out["latency_ms"]["p50"] == 300
    assert out["latency_ms"]["p95"] == 500


def test_token_totals_and_missing_usage_counted():
    runs = [run("ok", 1, 100, 20), run("ok", 1, 50, 10), run("ok", 1, None, None)]
    out = summarise(runs)
    assert out["tokens"] == {"input": 150, "output": 30}
    assert out["runs_without_usage"] == 1


def test_no_cost_without_explicit_prices():
    out = summarise([run("ok", 1, 1000, 1000)])
    assert out["estimated_cost_usd"] is None


def test_cost_uses_supplied_prices_only():
    out = summarise([run("ok", 1, 2000, 1000)], price_in_per_1k=0.001, price_out_per_1k=0.002)
    assert out["estimated_cost_usd"] == pytest.approx(0.004)


def test_get_run_rows_roundtrip(conn):
    rid = db.start_run(conn, "q", None)
    db.finish_run(conn, rid, "ok", "a", 0, 42, 7, 3)
    rows = db.get_run_rows(conn)
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["total_latency_ms"] == 42
    assert rows[0]["input_tokens"] == 7
