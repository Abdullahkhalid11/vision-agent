"""Runner helpers with a fake HTTP client. Run: python -m pytest tests/test_eval_run.py -v"""
import httpx

from eval.run_eval import build_report, post_with_retry


class FakeResponse:
    def __init__(self, status):
        self.status_code = status
        self.headers = {"content-type": "application/json"}


class FakeClient:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def post(self, url, **kwargs):
        self.calls += 1
        return FakeResponse(self.statuses.pop(0))


def test_retries_on_429_then_succeeds():
    client = FakeClient([429, 429, 200])
    waits = []
    r, retries, _ = post_with_retry(client, "/ask", sleep=waits.append, json={})
    assert r.status_code == 200
    assert retries == 2
    assert waits == [3.5, 3.5]


def test_gives_up_after_max_retries():
    client = FakeClient([429] * 10)
    r, retries, _ = post_with_retry(client, "/ask", sleep=lambda s: None, json={})
    assert r.status_code == 429
    assert retries == 5


def test_does_not_retry_other_errors():
    client = FakeClient([400])
    r, retries, _ = post_with_retry(client, "/ask", sleep=lambda s: None, json={})
    assert r.status_code == 400
    assert retries == 0
    assert client.calls == 1


def test_report_lists_failures_and_manual_rows():
    results = [
        {"id": "c1", "type": "count", "question": "q", "expected": 8, "answer": "7 cars",
         "verdict": "fail", "reason": "numbers [7], expected 8", "run_id": "r1",
         "latency_ms": 900.0, "retries": 0,
         "trace": [{"step": 1, "tool_name": "count_labels", "ok": 1, "error": None}]},
        {"id": "b1", "type": "ambiguous", "question": "q", "expected": "yes", "answer": "yes",
         "verdict": "manual", "reason": "", "run_id": "r2", "latency_ms": 700.0,
         "retries": 1, "trace": []},
    ]
    report = build_report(results)
    assert "Automatic: 0 pass, 1 fail of 1." in report
    assert "Manual rows awaiting a human verdict: 1." in report
    assert "### c1 (count)" in report
    assert "### b1 (ambiguous)" in report
    assert "Verdict: ______" in report
    assert "Rate-limit retries: 1." in report
