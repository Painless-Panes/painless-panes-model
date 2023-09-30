from ultralytics import YOLO

model = YOLO("window-model-nano-v1.pt")

results = model(
    f"val/images/D_c_31.75x45.jpeg",
    show=True,
    conf=0.25,
    save=True,
    project=".",
)
