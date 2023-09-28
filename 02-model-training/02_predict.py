from ultralytics import YOLO

model = YOLO("custom-model.pt")

filename = "example3"
results = model(
    f"test/{filename}.jpg",
    show=True,
    conf=0.25,
    save=True,
    project=".",
)
