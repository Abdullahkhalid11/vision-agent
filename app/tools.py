"""Agent tools. They read cached detections from SQLite; they never run the model.

Every tool has the signature tool(conn, image_id, **model_args). The agent loop injects
conn and image_id, so the model cannot choose which image a tool reads.
"""
from app import db

MIN_CONF_LOW = 0.1
MIN_CONF_HIGH = 0.9
DEFAULT_MIN_CONF = 0.25
MAX_LABEL_LEN = 40


class NoDetectionsError(LookupError):
    """The image is unknown, or it has not been run through /detect."""


def _check_min_confidence(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("min_confidence must be a number")
    if not MIN_CONF_LOW <= value <= MIN_CONF_HIGH:
        raise ValueError(f"min_confidence must be between {MIN_CONF_LOW} and {MIN_CONF_HIGH}")
    return float(value)


def _require_cached(conn, image_id):
    if db.get_image(conn, image_id) is None or not db.has_detections(conn, image_id):
        raise NoDetectionsError("no cached detections for this image; upload it with /detect first")


def as_api_detection(row):
    """Shape a DB row like the /detect response: label, confidence, bbox [x1, y1, x2, y2]."""
    return {
        "label": row["label"],
        "confidence": row["confidence"],
        "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
    }


def detect_objects(conn, image_id, min_confidence=DEFAULT_MIN_CONF):
    min_confidence = _check_min_confidence(min_confidence)
    _require_cached(conn, image_id)
    rows = db.get_detections(conn, image_id, min_confidence=min_confidence)
    return [as_api_detection(r) for r in rows]


def count_labels(conn, image_id, label, min_confidence=DEFAULT_MIN_CONF):
    min_confidence = _check_min_confidence(min_confidence)
    if not isinstance(label, str) or not label.strip() or len(label.strip()) > MAX_LABEL_LEN:
        raise ValueError(f"label must be a non-empty string of at most {MAX_LABEL_LEN} characters")
    normalised = label.strip().lower()
    _require_cached(conn, image_id)
    rows = db.get_detections(conn, image_id, min_confidence=min_confidence)
    count = sum(1 for r in rows if r["label"] == normalised)
    return {"label": normalised, "count": count, "min_confidence": min_confidence}


TOOLS = {"detect_objects": detect_objects, "count_labels": count_labels}

# Schemas the model sees. image_id and conn are injected by the loop and never appear here.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "detect_objects",
            "description": (
                "List the objects detected in the uploaded image, with confidence and bounding box. "
                "Use this to see what is present."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence from 0.1 to 0.9. Default 0.25.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_labels",
            "description": "Count detections of one COCO class in the uploaded image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "COCO class name, for example car, person, bicycle.",
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence from 0.1 to 0.9. Default 0.25.",
                    },
                },
                "required": ["label"],
            },
        },
    },
]
