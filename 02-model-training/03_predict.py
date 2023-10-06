from ultralytics import YOLO

model = YOLO("window-model-nano-v3.pt")

results = model(
    f"val/images/G_a_24x53-55x52.63-24x53.jpeg",
    show=True,
    conf=0.25,
    save=True,
    project=".",
)
