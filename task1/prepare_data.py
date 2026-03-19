#!/usr/bin/env python3
"""Convert COCO annotations to YOLO format and set up dataset."""
import json, os, shutil
from pathlib import Path

TRAIN_DIR   = "/Users/claudeagent/Downloads/train-2"
ANN_FILE    = f"{TRAIN_DIR}/annotations.json"
OUT_DIR     = Path(__file__).parent / "dataset"

with open(ANN_FILE) as f:
    coco = json.load(f)

# Build lookups
img_lookup = {img["id"]: img for img in coco["images"]}
cat_lookup = {cat["id"]: i for i, cat in enumerate(coco["categories"])}  # cat_id -> yolo_id
n_classes   = len(coco["categories"])
class_names = [c["name"] for c in coco["categories"]]

# Group annotations by image
from collections import defaultdict
img_anns = defaultdict(list)
for ann in coco["annotations"]:
    img_anns[ann["image_id"]].append(ann)

# Convert to YOLO format
imgs_out   = OUT_DIR / "images/train"
labels_out = OUT_DIR / "labels/train"
imgs_out.mkdir(parents=True, exist_ok=True)
labels_out.mkdir(parents=True, exist_ok=True)

for img_id, img_info in img_lookup.items():
    fname  = img_info["file_name"]
    src    = Path(TRAIN_DIR) / "images" / fname
    if not src.exists():
        continue
    shutil.copy(src, imgs_out / fname)

    W, H = img_info["width"], img_info["height"]
    lines = []
    for ann in img_anns[img_id]:
        yolo_cls = cat_lookup[ann["category_id"]]
        x, y, w, h = ann["bbox"]
        # YOLO: cx cy w h normalized
        cx = (x + w/2) / W
        cy = (y + h/2) / H
        nw = w / W
        nh = h / H
        lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    label_file = labels_out / (Path(fname).stem + ".txt")
    label_file.write_text("\n".join(lines))

# Write dataset yaml
yaml = f"""path: {OUT_DIR.absolute()}
train: images/train
val: images/train

nc: {n_classes}
names: {class_names}
"""
(OUT_DIR / "dataset.yaml").write_text(yaml)

# Save class names for inference
with open(Path(__file__).parent / "class_names.json", "w") as f:
    json.dump(class_names, f, ensure_ascii=False)

print(f"Done. {len(img_lookup)} images, {n_classes} classes, {len(coco['annotations'])} annotations")
print(f"Dataset yaml: {OUT_DIR}/dataset.yaml")
