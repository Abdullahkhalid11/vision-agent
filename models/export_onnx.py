"""Export YOLOv8n to ONNX. Run once from the repo root."""
from pathlib import Path
import shutil
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # downloads the pretrained weights on first run
exported = model.export(format="onnx", imgsz=640)  # returns the path of the .onnx file

dest = Path("models") / "yolov8n.onnx"
shutil.move(str(exported), dest)
print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
print("Classes:", len(model.names))