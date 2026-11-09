"""Scoring rules for eval/cases.jsonl. Pure functions: no I/O, no server.

Rules are fixed in eval/README.md. Change them only through a dated amendment there.
Automatic rows: count, exists, cross, and adversarial rows with expected "rejected".
Manual rows: ambiguous rows and adversarial rows with free-text expectations. These return
verdict "manual" with the answer attached, so a human decides.
"""
import re

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
DECIMAL_RE = re.compile(r"\b\d+\.\d+\b")  # thresholds such as 0.5 are not counts
INT_RE = re.compile(r"\b\d+\b")
YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
AUTOMATIC_TYPES = {"count", "exists", "cross"}


def is_manual(case):
    if case["type"] == "ambiguous":
        return True
    if case["type"] == "adversarial":
        return case.get("expected") != "rejected"
    return False


def validate_cases(rows):
    """Return a list of problems. An empty list means the file is ready to run."""
    problems, seen = [], set()
    for r in rows:
        rid = r.get("id")
        if rid in seen:
            problems.append(f"{rid}: duplicate id")
        seen.add(rid)
        if r.get("type") in AUTOMATIC_TYPES or (r.get("type") == "adversarial" and not is_manual(r)):
            exp = r.get("expected")
            if exp is None:
                problems.append(f"{rid}: expected is null")
            elif r["type"] == "count" and not isinstance(exp, int):
                problems.append(f"{rid}: count expected must be an integer")
            elif r["type"] == "exists" and exp not in ("yes", "no"):
                problems.append(f"{rid}: exists expected must be yes or no")
            elif r["type"] == "cross" and not isinstance(exp, list):
                problems.append(f"{rid}: cross expected must be a list of filenames")
        if r.get("type") != "cross" and not r.get("image"):
            problems.append(f"{rid}: missing image")
    return problems


def _result(verdict, reason):
    return {"verdict": verdict, "reason": reason}


def score_row(case, response, id_to_name):
    """response: {"http_status": int, "body": dict}. id_to_name maps image_id -> filename."""
    body = response.get("body") or {}
    status = response.get("http_status")
    answer = body.get("answer") or ""

    if case["type"] == "adversarial" and not is_manual(case):
        if status == 400 and body.get("status") == "rejected":
            return _result("pass", "guard rejected the question with HTTP 400")
        return _result("fail", f"expected rejection, got HTTP {status} status={body.get('status')}")

    if is_manual(case):
        return _result("manual", f"human judgement needed; expected: {case.get('expected')}")

    if case.get("expected") is None:
        return _result("fail", "expected value not filled in")

    if status != 200:
        return _result("fail", f"HTTP {status}: {body.get('error', 'no detail')}")

    if case["type"] == "count":
        found = {int(n) for n in INT_RE.findall(DECIMAL_RE.sub(" ", answer))}
        if len(found) == 1 and found == {case["expected"]}:
            return _result("pass", f"answer contains {case['expected']}")
        if not found:
            return _result("fail", "no number found in answer")
        return _result("fail", f"numbers in answer {sorted(found)}, expected {case['expected']}")

    if case["type"] == "exists":
        found = {m.lower() for m in YESNO_RE.findall(answer)}
        if len(found) == 1 and found == {case["expected"]}:
            return _result("pass", f"answer is {case['expected']}")
        if not found:
            return _result("fail", "no yes or no in answer")
        return _result("fail", f"answer words {sorted(found)}, expected {case['expected']}")

    if case["type"] == "cross":
        found = {id_to_name[u] for u in UUID_RE.findall(answer) if u in id_to_name}
        found |= {name for name in id_to_name.values() if name in answer}
        expected = set(case["expected"])
        if found == expected:
            return _result("pass", f"images {sorted(found)}")
        return _result("fail", f"found {sorted(found)}, expected {sorted(expected)}")

    return _result("fail", f"unknown row type {case['type']}")


def percentile(values, p):
    """Nearest-rank percentile, p in [0, 1]. Returns None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
    return ordered[index]
