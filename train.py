from ultralytics import YOLO
import os

# Load pretrained YOLOv8n
model = YOLO("yolov8n.pt")

# Train
results = model.train(
    data="data.yaml",
    epochs=300,
    imgsz=640,
    batch=8,
    device="cpu",     # use "cpu" if GPU not available
)

print("Training Completed!")


 