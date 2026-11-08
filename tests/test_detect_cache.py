"""/detect sha256 cache. Needs the FastAPI app import chain (detector model and guards),
so run it on the machine that has models/yolov8n.onnx: python -m pytest tests/test_detect_cache.py -v
The model is replaced with a fake, so inference is not exercised here."""
import io
from contextlib import closing

from fastapi.testclient import TestClient
from PIL import Image

from app import db, main


def jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 40, 40)).save(buf, "JPEG")
    return buf.getvalue()


def test_second_upload_of_same_bytes_is_served_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "v.db"))
    monkeypatch.setattr(main, "API_KEY", "test-key")

    calls = {"n": 0}

    def fake_detect(path):
        calls["n"] += 1
        return [{"label": "car", "confidence": 0.9, "bbox": [1.0, 2.0, 3.0, 4.0]}]

    monkeypatch.setattr(main, "detect", fake_detect)

    client = TestClient(main.app)
    data = jpeg_bytes()
    headers = {"X-API-Key": "test-key"}
    files = {"file": ("a.jpg", data, "image/jpeg")}

    r1 = client.post("/detect", files=files, headers=headers)
    r2 = client.post("/detect", files=files, headers=headers)

    assert r1.status_code == 200 and r2.status_code == 200
    b1, b2 = r1.json(), r2.json()
    assert b1["cached"] is False
    assert b2["cached"] is True
    assert b1["image_id"] == b2["image_id"]
    assert calls["n"] == 1  # the model ran once
    assert b2["detections"] == [
        {"label": "car", "confidence": 0.9, "bbox": [1.0, 2.0, 3.0, 4.0]}
    ]

    with closing(db.connect(main.DB_PATH)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 1

