"""Re-create model.onnx from the source weights (pip install ultralytics onnx onnxslim)."""
import urllib.request
from ultralytics import YOLO

urllib.request.urlretrieve("https://huggingface.co/vineetsarpal/yolov11n-car-damage/resolve/main/best.pt", "best.pt")
YOLO("best.pt").export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)  # -> best.onnx
