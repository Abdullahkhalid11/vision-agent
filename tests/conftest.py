import pytest

from app import db


@pytest.fixture
def conn(tmp_path):
    path = tmp_path / "t.db"
    db.init_db(path)
    c = db.connect(path)
    yield c
    c.close()


@pytest.fixture
def image_id(conn):
    """An image with three cars (0.9, 0.8, 0.3) and one person (0.6)."""
    iid = db.insert_image(conn, "sha-cars", 640, 480)
    db.insert_detections(conn, iid, [
        {"label": "car", "confidence": 0.9, "x1": 1, "y1": 1, "x2": 5, "y2": 5},
        {"label": "car", "confidence": 0.8, "x1": 2, "y1": 2, "x2": 6, "y2": 6},
        {"label": "car", "confidence": 0.3, "x1": 3, "y1": 3, "x2": 7, "y2": 7},
        {"label": "person", "confidence": 0.6, "x1": 0, "y1": 0, "x2": 4, "y2": 9},
    ], "yolov8n-onnx-640")
    return iid
