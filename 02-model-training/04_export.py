from ultralytics import YOLO

model = YOLO("window-model-nano-v2.pt")

model.export(format="onnx")

