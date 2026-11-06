"""Minimal benchmark for yolov8n.onnx: inference latency and peak RSS."""
import time
import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image

MODEL_PATH = "models/yolov8n.onnx"
IMAGE_PATH = "data/images/test1.jpg"
WARMUP, RUNS = 10, 100


def load_input(path):
    img = Image.open(path).convert("RGB").resize((640, 640))
    x = np.asarray(img, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None, ...]


proc = psutil.Process()
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
x = load_input(IMAGE_PATH)  # preprocessing is done once, outside the timer
rss_after_load = proc.memory_info().rss / 1024**2

for _ in range(WARMUP):
    session.run(None, {input_name: x})

times, peak = [], rss_after_load
for _ in range(RUNS):
    t0 = time.perf_counter()
    session.run(None, {input_name: x})
    times.append((time.perf_counter() - t0) * 1000)
    peak = max(peak, proc.memory_info().rss / 1024**2)

p50, p95 = np.percentile(times, [50, 95])
print(f"Inference p50={p50:.1f} ms  p95={p95:.1f} ms  (preprocessing excluded)")
print(f"RSS after load={rss_after_load:.0f} MB  peak={peak:.0f} MB")
plan = "Starter (512 MB)" if peak < 350 else "Standard (2 GB)"
print("Plan candidate:", plan, "- laptop measurement, confirm on Render on Nov 10")