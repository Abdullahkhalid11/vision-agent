"""Tests for app/tools.py. Run: python -m pytest tests/test_tools.py -v"""
import pytest

from app import db
from app.tools import TOOL_SCHEMAS, NoDetectionsError, count_labels, detect_objects


def test_detect_objects_filters_and_formats(conn, image_id):
    out = detect_objects(conn, image_id, min_confidence=0.5)
    assert [d["label"] for d in out] == ["car", "car", "person"]
    assert set(out[0]) == {"label", "confidence", "bbox"}
    assert out[0]["bbox"] == [1, 1, 5, 5]


def test_count_is_case_insensitive(conn, image_id):
    assert count_labels(conn, image_id, "CAR") == {
        "label": "car", "count": 3, "min_confidence": 0.25}


def test_count_respects_min_confidence(conn, image_id):
    assert count_labels(conn, image_id, "car", min_confidence=0.5)["count"] == 2


def test_count_zero_for_absent_label(conn, image_id):
    assert count_labels(conn, image_id, "bicycle")["count"] == 0


@pytest.mark.parametrize("bad", [0.05, 0.95, True, "0.5", None])
def test_min_confidence_rejected(conn, image_id, bad):
    with pytest.raises(ValueError):
        detect_objects(conn, image_id, min_confidence=bad)


@pytest.mark.parametrize("bad", ["", "   ", "x" * 41, 7, None])
def test_label_rejected(conn, image_id, bad):
    with pytest.raises(ValueError):
        count_labels(conn, image_id, bad)


def test_unknown_image_raises_no_detections(conn):
    with pytest.raises(NoDetectionsError):
        detect_objects(conn, "does-not-exist")
    with pytest.raises(NoDetectionsError):
        count_labels(conn, "does-not-exist", "car")


def test_image_without_detections_raises(conn):
    empty_id = db.insert_image(conn, "sha-empty", 10, 10)
    with pytest.raises(NoDetectionsError):
        count_labels(conn, empty_id, "car")


def test_schemas_do_not_expose_image_fields():
    for schema in TOOL_SCHEMAS:
        props = schema["function"]["parameters"]["properties"]
        assert "image_id" not in props
        assert "image_path" not in props
