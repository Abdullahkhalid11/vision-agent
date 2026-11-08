"""Tools the agent may call. Each returns plain JSON-serialisable data."""
from app.detector import detect


def detect_objects(image_path, min_confidence=0.25):
    return [d for d in detect(image_path) if d["confidence"] >= min_confidence]


def count_labels(detections, label):
    return {"label": label, "count": sum(1 for d in detections if d["label"] == label)}


TOOLS = {"detect_objects": detect_objects, "count_labels": count_labels}