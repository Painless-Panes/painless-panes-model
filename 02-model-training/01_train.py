from ultralytics import YOLO

# model = YOLO('yolov8n.pt')
# model = YOLO('window-model-nano-v1.pt')
model = YOLO('runs/train/weights/last.pt')
# model = YOLO('yolov8m.pt')

model.train(
    data='annotated_data.yaml',
    epochs=200,
    imgsz=640,
    project='./runs',
    resume=True,
)
