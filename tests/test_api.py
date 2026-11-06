from fastapi.testclient import TestClient

from app.main import API_KEY, app

client = TestClient(app)
GOOD = "data/images/test1.jpg"
H = {"X-API-Key": API_KEY or ""}


def post(content, key=H, name="a.jpg", ctype="image/jpeg"):
    return client.post("/detect", headers=key, files={"file": (name, content, ctype)})


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_detect_requires_key():
    with open(GOOD, "rb") as f:
        assert post(f.read(), key={}).status_code == 401


def test_detect_wrong_key():
    with open(GOOD, "rb") as f:
        assert post(f.read(), key={"X-API-Key": "wrong"}).status_code == 401


def test_detect_rejects_text_renamed_jpg():
    assert post(b"hello world").status_code == 415


def test_detect_rejects_corrupt_jpeg():
    assert post(b"\xff\xd8\xff" + b"0" * 100).status_code == 400


def test_detect_rejects_oversize():
    big = b"\xff\xd8\xff" + b"0" * (5 * 1024 * 1024)
    assert post(big).status_code == 413


def test_detect_valid_image():
    with open(GOOD, "rb") as f:
        r = post(f.read())
    assert r.status_code == 200
    body = r.json()
    assert len(body["sha256"]) == 64
    assert isinstance(body["detections"], list)