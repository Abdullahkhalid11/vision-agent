"""Compare ONNX detector with PyTorch on one image. Usage: python -m bench.compare <image>"""
import sys

from ultralytics import YOLO

from app.detector import detect

THRESH = 0.5  # compare confident detections; low-confidence boxes are noise

path = sys.argv[1]
onnx = [(d["label"], d["confidence"]) for d in detect(path)]
res = YOLO("yolov8n.pt")(path, verbose=False)[0]
pt = [(res.names[int(c)], round(float(s), 3)) for c, s in zip(res.boxes.cls, res.boxes.conf)]


def strong(dets):
    return sorted(label for label, conf in dets if conf >= THRESH)


print("ONNX:", sorted(onnx, key=lambda t: -t[1]))
print("PyTorch:", sorted(pt, key=lambda t: -t[1]))
print("ONNX count:", len(onnx), "| PyTorch count:", len(pt))
print("Confident (>=0.5) labels match:", strong(onnx) == strong(pt))