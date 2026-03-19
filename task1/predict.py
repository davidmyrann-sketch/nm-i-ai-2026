#!/usr/bin/env python3
"""
Inference script for NM i AI 2026 — NorgesGruppen Object Detection.
Usage: python predict.py --image <path> [--model <weights.pt>]
Output: JSON with detections in COCO format
"""
import json, sys, argparse
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).parent

def load_model(weights=None):
    if weights is None:
        # Find best weights from training run
        candidates = sorted(BASE.glob("runs/ngd_yolov8m/weights/best.pt"))
        if not candidates:
            candidates = sorted(BASE.glob("runs/**/best.pt"), recursive=True) if not candidates else candidates
        weights = str(candidates[0]) if candidates else "yolov8m.pt"
    return YOLO(weights)

def predict(image_path, model, conf=0.25, iou=0.45):
    with open(BASE / "class_names.json") as f:
        class_names = json.load(f)

    results = model(image_path, conf=conf, iou=iou, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO: x,y,w,h
            "category_id": int(box.cls[0].item()),
            "category_name": class_names[int(box.cls[0].item())],
            "score": float(box.conf[0].item())
        })
    return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    model = load_model(args.model)
    dets  = predict(args.image, model, conf=args.conf)
    print(json.dumps(dets, indent=2))
