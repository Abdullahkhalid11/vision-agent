"""Agent tools. They read cached detections from SQLite; they never run the model.

Every tool has the signature tool(conn, image_id, **model_args). The agent loop injects
conn and image_id, so the model cannot choose which image a tool reads.
search_events is the one cross-image tool. It reads metadata counts only, never image data.
"""
import re
from datetime import date

from app import db

MIN_CONF_LOW = 0.1
MIN_CONF_HIGH = 0.9
DEFAULT_MIN_CONF = 0.25
MAX_LABEL_LEN = 40
MAX_LIMIT = 50
DEFAULT_LIMIT = 20
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class NoDetectionsError(LookupError):
    """The image is unknown, or it has not been run through /detect."""


def _check_min_confidence(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("min_confidence must be a number")
    if not MIN_CONF_LOW <= value <= MIN_CONF_HIGH:
        raise ValueError(f"min_confidence must be between {MIN_CONF_LOW} and {MIN_CONF_HIGH}")
    return float(value)


def _check_label(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > MAX_LABEL_LEN:
        raise ValueError(f"label must be a non-empty string of at most {MAX_LABEL_LEN} characters")
    return value.strip().lower()


def _check_date(value, name):
    if not isinstance(value, str) or not _DATE_RE.fullmatch(value):
        raise ValueError(f"{name} must be a date in YYYY-MM-DD form")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise ValueError(f"{name} is not a valid calendar date")


def _check_limit(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_LIMIT:
        raise ValueError(f"limit must be an integer from 1 to {MAX_LIMIT}")
    return value


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
    normalised = _check_label(label)
    _require_cached(conn, image_id)
    rows = db.get_detections(conn, image_id, min_confidence=min_confidence)
    count = sum(1 for r in rows if r["label"] == normalised)
    return {"label": normalised, "count": count, "min_confidence": min_confidence}


def search_events(conn, image_id, label, since, until,
                  min_confidence=DEFAULT_MIN_CONF, limit=DEFAULT_LIMIT):
    """Images with the label in a UTC date range, newest first.

    image_id is part of the uniform tool signature but is not used: this search is
    cross-image and reads metadata only.
    """
    normalised = _check_label(label)
    since_d = _check_date(since, "since")
    until_d = _check_date(until, "until")
    if since_d > until_d:
        raise ValueError("since must not be after until")
    min_confidence = _check_min_confidence(min_confidence)
    limit = _check_limit(limit)
    return db.find_images_by_label(conn, normalised, since_d, until_d, min_confidence, limit)


TOOLS = {
    "detect_objects": detect_objects,
    "count_labels": count_labels,
    "search_events": search_events,
}

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
    {
        "type": "function",
        "function": {
            "name": "search_events",
            "description": (
                "Find previously uploaded images that contain a COCO class, uploaded within a "
                "date range. Use for questions across images, such as 'this week'. "
                "Returns image ids, upload times, and counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "COCO class name, for example car, person.",
                    },
                    "since": {
                        "type": "string",
                        "description": "Start date, inclusive, in YYYY-MM-DD (UTC).",
                    },
                    "until": {
                        "type": "string",
                        "description": "End date, inclusive, in YYYY-MM-DD (UTC).",
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence from 0.1 to 0.9. Default 0.25.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results, 1 to 50. Default 20.",
                    },
                },
                "required": ["label", "since", "until"],
            },
        },
    },
]
