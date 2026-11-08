"""FastAPI service: /health and /detect. Auth and guards here; inference in app.detector;
storage in app.db. Uploads are never kept on disk, only their hash, size and detections."""
import hashlib
import hmac
import io
import os
import tempfile
from contextlib import closing
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from app import db
from app.ask import handle_question
from app.detector import MODEL_PATH, detect
from app.guards import MAX_BYTES, RejectedUpload, check_upload
from app.llm import groq_llm
from app.rate_limit import TokenBucketLimiter
from app.tools import as_api_detection

load_dotenv()
API_KEY = os.getenv("API_KEY")
DB_PATH = os.getenv("VISION_DB_PATH", "data/vision.db")
MODEL_VERSION = "yolov8n-onnx-640"

app = FastAPI(title="Vision Agent", version="0.4")

limiter = TokenBucketLimiter()  # 20 requests per minute per API key


def rate_limit(x_api_key: str | None = Header(default=None)):
    if not limiter.allow(x_api_key or "anonymous"):
        raise HTTPException(429, "rate limit exceeded; try again shortly")


def require_api_key(x_api_key: str | None = Header(default=None)):
    if not API_KEY:
        raise HTTPException(500, "server API_KEY is not configured")
    if x_api_key is None or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "invalid API key")


@app.get("/health")
def health():
    return {"status": "ok", "model_file_present": Path(MODEL_PATH).exists()}


def _store(sha256, width, height, detections):
    rows = [
        {
            "label": d["label"],
            "confidence": d["confidence"],
            "x1": d["bbox"][0], "y1": d["bbox"][1],
            "x2": d["bbox"][2], "y2": d["bbox"][3],
        }
        for d in detections
    ]
    db.init_db(DB_PATH)
    with closing(db.connect(DB_PATH)) as conn:
        image_id = db.insert_image(conn, sha256, width, height)
        db.insert_detections(conn, image_id, rows, MODEL_VERSION)
    return image_id


# Plain def, not async def: detect() is CPU-bound. FastAPI runs plain functions in a
# thread pool, so the server stays responsive.
@app.post("/detect", dependencies=[Depends(require_api_key)])
def detect_endpoint(file: UploadFile = File(...)):
    data = file.file.read(MAX_BYTES + 1)  # read at most one byte over the limit
    try:
        check_upload(data)
    except RejectedUpload as e:
        raise HTTPException(e.status, e.message)

    sha256 = hashlib.sha256(data).hexdigest()

    # Cache hit: same bytes seen before, with detections stored. Skip the model.
    db.init_db(DB_PATH)
    with closing(db.connect(DB_PATH)) as conn:
        image = db.get_image_by_sha(conn, sha256)
        if image is not None and db.has_detections(conn, image["id"]):
            rows = db.get_detections(conn, image["id"])
            return {
                "image_id": image["id"],
                "sha256": sha256,
                "cached": True,
                "detections": [as_api_detection(r) for r in rows],
            }

    # Cache miss: run the model on a temp file, then delete it.
    with Image.open(io.BytesIO(data)) as img:
        width, height = img.size
    suffix = ".jpg" if data.startswith(b"\xff\xd8\xff") else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        detections = detect(path)
    finally:
        os.remove(path)

    image_id = _store(sha256, width, height, detections)
    return {
        "image_id": image_id,
        "sha256": sha256,
        "cached": False,
        "detections": detections,
    }


class AskRequest(BaseModel):
    image_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=500)


@app.post("/ask", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def ask_endpoint(body: AskRequest):
    # groq_llm is looked up at call time so tests can replace it.
    db.init_db(DB_PATH)
    with closing(db.connect(DB_PATH)) as conn:
        status, payload = handle_question(conn, body.image_id, body.question, groq_llm)
    return JSONResponse(status_code=status, content=payload)
