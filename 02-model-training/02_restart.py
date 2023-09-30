""" Use this to restart a training that died
"""
from ultralytics import YOLO

model = YOLO('runs/train/weights/last.pt')

model.train(
    data='annotated_data.yaml',
    epochs=200,
    imgsz=640,
    project='./runs',
    resume=True,
)
