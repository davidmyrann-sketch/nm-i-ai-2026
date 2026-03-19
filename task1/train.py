#!/usr/bin/env python3
"""Train YOLOv8 on NorgesGruppen grocery dataset."""
from ultralytics import YOLO
from pathlib import Path

YAML = str(Path(__file__).parent / "dataset/dataset.yaml")

model = YOLO("yolov8m.pt")  # medium — good balance speed/accuracy

results = model.train(
    data=YAML,
    epochs=80,
    imgsz=640,
    batch=8,
    patience=20,
    lr0=0.01,
    augment=True,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    degrees=5.0,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    project=str(Path(__file__).parent / "runs"),
    name="ngd_yolov8m",
    exist_ok=True,
    device="cpu",
    workers=4,
)

print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"Model saved to: {results.save_dir}")
