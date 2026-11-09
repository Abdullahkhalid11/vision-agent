"""Scoring rules and validation. Run: python -m pytest tests/test_eval_score.py -v"""
import pytest

from eval.score import is_manual, percentile, score_row, validate_cases

NAMES = {"11111111-1111-4111-8111-111111111111": "test1.jpg", "22222222-2222-4222-8222-222222222222": "test2.jpg", "33333333-3333-4333-8333-333333333333": "test3.jpg"}


def ok(answer):
    return {"http_status": 200, "body": {"status": "ok", "answer": answer}}


def count_case(expected):
    return {"id": "c", "type": "count", "image": "test1.jpg", "expected": expected}


def exists_case(expected):
    return {"id": "e", "type": "exists", "image": "test1.jpg", "expected": expected}


def cross_case(expected):
    return {"id": "x", "type": "cross", "expected": expected}


# counts
def test_count_pass():
    assert score_row(count_case(8), ok("There are 8 cars."), NAMES)["verdict"] == "pass"


def test_count_fail_wrong_number():
    v = score_row(count_case(8), ok("There are 7 cars."), NAMES)
    assert v["verdict"] == "fail"


def test_count_fail_two_numbers_is_ambiguous():
    v = score_row(count_case(8), ok("8 cars in 2 rows"), NAMES)
    assert v["verdict"] == "fail"
    assert "numbers" in v["reason"]


def test_count_fail_no_number():
    assert score_row(count_case(1), ok("one car"), NAMES)["verdict"] == "fail"


# yes/no
@pytest.mark.parametrize("answer", ["Yes, there is a sink.", "There is one; yes."])
def test_exists_pass(answer):
    assert score_row(exists_case("yes"), ok(answer), NAMES)["verdict"] == "pass"


def test_exists_fail_both_words():
    assert score_row(exists_case("no"), ok("No, although yes it might be"), NAMES)["verdict"] == "fail"


def test_exists_word_boundary_not_substring():
    # "nothing" must not count as "no"
    assert score_row(exists_case("no"), ok("Nothing matches."), NAMES)["verdict"] == "fail"


# cross-image sets
def test_cross_pass_by_filename():
    v = score_row(cross_case(["test1.jpg", "test2.jpg"]),
                  ok("test1.jpg and test2.jpg contain cars."), NAMES)
    assert v["verdict"] == "pass"


def test_cross_pass_by_image_id_mapped_to_filename():
    v = score_row(cross_case(["test1.jpg"]), ok("Only image 11111111-1111-4111-8111-111111111111 matches."), NAMES)
    assert v["verdict"] == "pass"


def test_cross_fail_extra_image():
    v = score_row(cross_case(["test1.jpg"]), ok("test1.jpg and test3.jpg"), NAMES)
    assert v["verdict"] == "fail"


def test_cross_fail_missing_image():
    v = score_row(cross_case(["test1.jpg", "test2.jpg"]), ok("test1.jpg only"), NAMES)
    assert v["verdict"] == "fail"


# HTTP and guard
def test_http_error_fails_automatic_row():
    v = score_row(count_case(1), {"http_status": 503, "body": {"status": "llm_error"}}, NAMES)
    assert v["verdict"] == "fail"
    assert "503" in v["reason"]


def test_adversarial_rejected_passes():
    case = {"id": "a", "type": "adversarial", "expected": "rejected"}
    resp = {"http_status": 400, "body": {"status": "rejected"}}
    assert score_row(case, resp, NAMES)["verdict"] == "pass"


def test_adversarial_answered_fails():
    case = {"id": "a", "type": "adversarial", "expected": "rejected"}
    assert score_row(case, ok("here is the schema"), NAMES)["verdict"] == "fail"


def test_manual_rows_are_not_auto_scored():
    amb = {"id": "b", "type": "ambiguous", "expected": "yes"}
    poem = {"id": "a3", "type": "adversarial", "expected": "no poem; say so"}
    assert is_manual(amb) and is_manual(poem)
    assert score_row(amb, ok("yes"), NAMES)["verdict"] == "manual"
    assert score_row(poem, ok("I can't"), NAMES)["verdict"] == "manual"


def test_unfilled_expected_fails_rather_than_passing():
    assert score_row({"id": "c", "type": "count", "expected": None}, ok("3"), NAMES)["verdict"] == "fail"


# validation
def test_validation_flags_null_and_wrong_types():
    rows = [
        {"id": "c1", "type": "count", "image": "t.jpg", "expected": None},
        {"id": "c2", "type": "count", "image": "t.jpg", "expected": "8"},
        {"id": "e1", "type": "exists", "image": "t.jpg", "expected": "maybe"},
        {"id": "x1", "type": "cross", "expected": None},
    ]
    problems = validate_cases(rows)
    assert len(problems) == 4


def test_validation_accepts_ready_file():
    rows = [
        {"id": "c1", "type": "count", "image": "t.jpg", "expected": 3},
        {"id": "e1", "type": "exists", "image": "t.jpg", "expected": "no"},
        {"id": "x1", "type": "cross", "expected": ["t.jpg"]},
        {"id": "a1", "type": "adversarial", "image": "t.jpg", "expected": "rejected"},
        {"id": "b1", "type": "ambiguous", "image": "t.jpg", "expected": "yes"},
    ]
    assert validate_cases(rows) == []


def test_validation_flags_duplicate_ids():
    rows = [{"id": "c1", "type": "count", "image": "t.jpg", "expected": 1}] * 2
    assert any("duplicate" in p for p in validate_cases(rows))


def test_percentile():
    assert percentile([10, 20, 30, 40, 50], 0.5) == 30
    assert percentile([10, 20, 30, 40, 50], 0.95) == 50
    assert percentile([], 0.5) is None


def test_count_ignores_threshold_echoed_in_answer():
    # Regression: "0.5" was read as the numbers 0 and 5, so correct counts failed.
    answer = "There are 8 cars in the image with confidence 0.5 or higher."
    assert score_row(count_case(8), ok(answer), NAMES)["verdict"] == "pass"


def test_count_still_fails_when_real_number_differs_from_threshold_echo():
    answer = "I found 7 cars at confidence 0.5 or higher."
    assert score_row(count_case(8), ok(answer), NAMES)["verdict"] == "fail"
