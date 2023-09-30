from ultralytics import YOLO

# # for a total fresh start, start from one of the YOLO v8 models
# model = YOLO('yolov8n.pt')  

# otherwise, start from an existing model to improve it after adding to the
# training set
model = YOLO('window-model-nano-v2.pt')

model.train(
    data='annotated_data.yaml',
    epochs=200,
    imgsz=640,
    project='./runs',
)
