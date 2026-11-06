"""YOLOv8n ONNX detector. Loads the model once. No HTTP or LLM code here."""
import ast

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "models/yolov8n.onnx"
IMG_SIZE = 640
CONF_MIN = 0.25
IOU_MAX = 0.7

_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name
_names = ast.literal_eval(_session.get_modelmeta().custom_metadata_map["names"])


def _iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def letterbox(img):
    """Scale to fit IMG_SIZE without distortion, then pad. Returns canvas, scale, pad offsets."""
    w0, h0 = img.size
    r = min(IMG_SIZE / w0, IMG_SIZE / h0)
    nw, nh = round(w0 * r), round(h0 * r)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (114, 114, 114))
    padx, pady = (IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (padx, pady))
    return canvas, r, padx, pady


def detect(image_path):
    """Return a list of {label, confidence, bbox} in original image pixels."""
    img = Image.open(image_path).convert("RGB")
    w0, h0 = img.size
    canvas, r, padx, pady = letterbox(img)
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None, ...]

    out = _session.run(None, {_input_name: x})[0][0].T  # (8400, 84)
    cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    scores = out[:, 4:]
    cls = scores.argmax(axis=1)
    conf = scores.max(axis=1)

    keep = conf > CONF_MIN
    cls, conf = cls[keep], conf[keep]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)[keep]

    # Per-class non-maximum suppression.
    selected = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        idx = idx[np.argsort(-conf[idx])]
        while idx.size:
            i = idx[0]
            selected.append(i)
            idx = idx[1:][_iou(xyxy[i], xyxy[idx[1:]]) < IOU_MAX]

    results = []
    for i in sorted(selected, key=lambda i: -conf[i]):
        x1, y1, x2, y2 = xyxy[i]
        # Undo the letterbox: remove padding, divide by scale, clip to the image.
        x1 = np.clip((x1 - padx) / r, 0, w0)
        y1 = np.clip((y1 - pady) / r, 0, h0)
        x2 = np.clip((x2 - padx) / r, 0, w0)
        y2 = np.clip((y2 - pady) / r, 0, h0)
        results.append({
            "label": _names[int(cls[i])],
            "confidence": round(float(conf[i]), 3),
            "bbox": [round(float(x1), 1), round(float(y1), 1),
                     round(float(x2), 1), round(float(y2), 1)],
        })
    return results


if __name__ == "__main__":
    import sys
    for d in detect(sys.argv[1]):
        print(d)