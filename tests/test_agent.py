"""Agent loop tests with a scripted fake LLM. No network, no model. Run:
python -m pytest tests/test_agent.py -v"""
import json

from app import db
from app.agent import MAX_STEPS, run_agent


class Scripted:
    """Fake LLM. Pops one response per call and records the messages it was given."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    def __call__(self, messages):
        self.seen.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("fake LLM ran out of responses")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def call(name, args, call_id):
    return {
        "tool_calls": [{"id": call_id, "name": name, "args": args}],
        "raw": {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]},
    }


def text(answer, usage=None):
    out = {"text": answer, "raw": {"role": "assistant", "content": answer}}
    if usage is not None:
        out["usage"] = usage
    return out


def test_happy_path_persists_run_and_tool_calls(conn, image_id):
    llm = Scripted([
        call("detect_objects", {}, "c1"),
        call("count_labels", {"label": "car"}, "c2"),
        text("There are 3 cars."),
    ])
    res = run_agent(conn, "How many cars?", image_id, llm)

    assert res["status"] == "ok"
    assert res["answer"] == "There are 3 cars."
    assert res["steps"] == 2

    run = db.get_run(conn, res["run_id"])
    assert run["status"] == "ok"
    assert run["steps"] == 2
    assert run["image_id"] == image_id
    assert run["total_latency_ms"] is not None

    calls = db.get_tool_calls(conn, res["run_id"])
    assert [c["tool_name"] for c in calls] == ["detect_objects", "count_labels"]
    assert [c["step"] for c in calls] == [1, 2]
    assert all(c["ok"] == 1 for c in calls)
    assert '"count": 3' in calls[1]["result_summary"]


def test_tool_results_are_wrapped_as_data(conn, image_id):
    llm = Scripted([call("count_labels", {"label": "car"}, "c1"), text("3")])
    run_agent(conn, "q", image_id, llm)
    tool_msg = llm.seen[-1][-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "c1"
    assert tool_msg["content"].startswith(
        '<tool_result name="count_labels" ok="true" trust="data-only">')
    assert tool_msg["content"].rstrip().endswith("</tool_result>")


def test_count_without_detections_recorded_as_error_and_run_continues(conn):
    empty_id = db.insert_image(conn, "sha-empty", 10, 10)
    llm = Scripted([call("count_labels", {"label": "car"}, "c1"),
                    text("I could not count the cars.")])
    res = run_agent(conn, "q", empty_id, llm)

    assert res["status"] == "ok"
    calls = db.get_tool_calls(conn, res["run_id"])
    assert calls[0]["ok"] == 0
    assert "no cached detections" in calls[0]["error"]


def test_model_cannot_override_image_id(conn, image_id):
    llm = Scripted([call("count_labels",
                         {"label": "car", "image_id": "someone-else"}, "c1"),
                    text("done")])
    res = run_agent(conn, "q", image_id, llm)

    calls = db.get_tool_calls(conn, res["run_id"])
    assert calls[0]["ok"] == 0
    assert "image_id" in calls[0]["error"]


def test_unknown_tool_recorded_and_not_executed(conn, image_id):
    llm = Scripted([call("delete_everything", {}, "c1"), text("ok")])
    res = run_agent(conn, "q", image_id, llm)

    calls = db.get_tool_calls(conn, res["run_id"])
    assert calls[0]["tool_name"] == "delete_everything"
    assert calls[0]["ok"] == 0
    assert calls[0]["error"] == "unknown tool"


def test_invalid_json_arguments_recorded(conn, image_id):
    bad = {"tool_calls": [{"id": "c1", "name": "count_labels", "args": None}],
           "raw": {"role": "assistant", "content": ""}}
    llm = Scripted([bad, text("ok")])
    res = run_agent(conn, "q", image_id, llm)

    calls = db.get_tool_calls(conn, res["run_id"])
    assert calls[0]["ok"] == 0
    assert calls[0]["error"] == "invalid JSON arguments"


def test_max_steps_enforced(conn, image_id):
    llm = Scripted([call("detect_objects", {}, f"c{i}") for i in range(MAX_STEPS + 1)])
    res = run_agent(conn, "loop forever", image_id, llm)

    assert res["status"] == "max_steps"
    assert res["steps"] == MAX_STEPS
    assert res["answer"] is None
    assert len(db.get_tool_calls(conn, res["run_id"])) == MAX_STEPS
    assert db.get_run(conn, res["run_id"])["status"] == "max_steps"


def test_llm_error_persisted_with_detail_but_returned_generically(conn, image_id):
    llm = Scripted([RuntimeError("upstream timeout after 15s")])
    res = run_agent(conn, "q", image_id, llm)

    assert res["status"] == "llm_error"
    assert res["error"] == "RuntimeError"  # caller sees type only
    run = db.get_run(conn, res["run_id"])
    assert run["status"] == "llm_error"
    assert "upstream timeout" in run["error"]  # full detail stays in the DB
    assert run["answer"] is None


def test_token_usage_summed_across_rounds(conn, image_id):
    llm = Scripted([
        {**call("detect_objects", {}, "c1"),
         "usage": {"input_tokens": 100, "output_tokens": 10}},
        text("done", usage={"input_tokens": 150, "output_tokens": 25}),
    ])
    res = run_agent(conn, "q", image_id, llm)

    run = db.get_run(conn, res["run_id"])
    assert run["input_tokens"] == 250
    assert run["output_tokens"] == 35


def test_missing_usage_stores_null_tokens(conn, image_id):
    llm = Scripted([text("no usage reported")])
    res = run_agent(conn, "q", image_id, llm)
    run = db.get_run(conn, res["run_id"])
    assert run["input_tokens"] is None
    assert run["output_tokens"] is None
