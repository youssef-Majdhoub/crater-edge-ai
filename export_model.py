
from ultralytics import YOLO

# Load the lightweight baseline model
model = YOLO("yolov8n.pt")

# Export for ONNX Runtime inference
model.export(
    format="onnx",
    imgsz=640,
    simplify=True,
    dynamic=False,
    nms=False,
)

print("ONNX export complete.")
