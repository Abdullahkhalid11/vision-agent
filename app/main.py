"""FastAPI service: /health and /detect. Auth and guards here; detection logic in app.detector."""
import hashlib
import hmac
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile

from app.detector import MODEL_PATH, detect
from app.guards import MAX_BYTES, RejectedUpload, check_upload

load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI(title="Vision Agent", version="0.2")


def require_api_key(x_api_key: str | None = Header(default=None)):
    if not API_KEY:
        raise HTTPException(500, "server API_KEY is not configured")
    if x_api_key is None or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "invalid API key")


@app.get("/health")
def health():
    return {"status": "ok", "model_file_present": Path(MODEL_PATH).exists()}


# Plain def, not async def: detect() is CPU-bound. FastAPI runs plain functions
# in a thread pool, so the server stays responsive.
@app.post("/detect", dependencies=[Depends(require_api_key)])
def detect_endpoint(file: UploadFile = File(...)):
    data = file.file.read(MAX_BYTES + 1)  # read at most one byte over the limit
    try:
        check_upload(data)
    except RejectedUpload as e:
        raise HTTPException(e.status, e.message)

    sha256 = hashlib.sha256(data).hexdigest()
    suffix = ".jpg" if data.startswith(b"\xff\xd8\xff") else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        detections = detect(path)
    finally:
        os.remove(path)  # uploads are never kept on disk
    return {"sha256": sha256, "detections": detections}