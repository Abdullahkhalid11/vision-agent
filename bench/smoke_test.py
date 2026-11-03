"""Smoke test: run yolov8n.onnx on one image and print output shape and confident anchors."""
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "models/yolov8n.onnx"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25

image_path = sys.argv[1]

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_meta = session.get_inputs()[0]
print("Input:", input_meta.name, input_meta.shape)

img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
x = np.asarray(img, dtype=np.float32) / 255.0
x = x.transpose(2, 0, 1)[None, ...]  # HWC -> NCHW with batch dim

outputs = session.run(None, {input_meta.name: x})[0]
print("Output shape:", outputs.shape)  # expected (1, 84, 8400)

class_scores = outputs[0, 4:, :]       # (80, 8400)
best_per_anchor = class_scores.max(axis=0)
print("Anchors with confidence >", CONF_THRESHOLD, ":", int((best_per_anchor > CONF_THRESHOLD).sum()))
print("Highest confidence:", round(float(best_per_anchor.max()), 3))