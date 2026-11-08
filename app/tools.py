"""Tools the agent may call. Each returns plain JSON-serialisable data."""
from app.detector import detect


def detect_objects(image_path, min_confidence=0.25):
    if not 0.1 <= min_confidence <= 0.9:
        raise ValueError("min_confidence must be between 0.1 and 0.9")
    return [d for d in detect(image_path) if d["confidence"] >= min_confidence]


def count_labels(detections, label):
    return {"label": label, "count": sum(1 for d in detections if d["label"] == label)}


TOOLS = {"detect_objects": detect_objects, "count_labels": count_labels}

# Schemas the model sees. image_path is injected by the loop, so the model never supplies it.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "detect_objects",
            "description": "Run the object detector on the uploaded image. Call this before counting.",
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
            "description": "Count detections of one COCO class in the most recent detect_objects result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "COCO class name, for example car, person, bicycle.",
                    }
                },
                "required": ["label"],
            },
        },
    },
]