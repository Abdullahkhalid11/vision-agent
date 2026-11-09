"""HTTP test for GET /analytics. Needs the app import chain, so run on your machine:
python -m pytest tests/test_analytics_api.py -v"""
from contextlib import closing

from fastapi.testclient import TestClient

from app import db, main


def test_analytics_requires_key_and_reports_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setattr(main, "API_KEY", "test-key")
    db.init_db(main.DB_PATH)
    with closing(db.connect(main.DB_PATH)) as conn:
        rid = db.start_run(conn, "q", None)
        db.finish_run(conn, rid, "ok", "a", 1, 120, 50, 10)
    client = TestClient(main.app)

    assert client.get("/analytics").status_code == 401
    r = client.get("/analytics", headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    body = r.json()
    assert body["runs"] == 1
    assert body["tokens"] == {"input": 50, "output": 10}
    assert body["estimated_cost_usd"] is None
