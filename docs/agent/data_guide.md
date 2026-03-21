# Data Guide

## Classes
| ID | Name | Description |
|----|------|-------------|
| 0 | screw | Normal screw present |
| 1 | missing_screw | Empty screw position |
| 2 | missing_component | Missing structural component |

## Dataset Structure
```
data/processed/phase1_multiclass_v1/
  train/images/  (717 images)
  train/labels/  (YOLO format txt)
  val/images/    (89 images)
  val/labels/
  test/images/   (91 images)
  test/labels/
  data.yaml
```

## Label Format (YOLO)
Each `.txt` file: `class_id cx cy w h` (normalized 0-1)

Example: `0 0.432 0.567 0.089 0.112` = screw at (43.2%, 56.7%) with 8.9% x 11.2% size

## Weak Labels (CRITICAL)
- Weak label = bbox covers >70% of image (e.g., `0.5 0.5 0.8 0.8`)
- Weak labels poison training (V2 mAP dropped from 0.99 to 0.84)
- Use `label_validator.py --check-only` before training
- Muhammet must provide labels WITH augmented images

## Data Sources
- `erdogan1/`, `erdogan2/` — Factory raw images
- `roboflowetiketlenen/` — Roboflow COCO-labeled (gold standard)
- `coklanmis/`, `coklanmisacili/` — Augmented images (some unlabeled!)

## Augmentation Rules
1. Labels must transform WITH images (rotate -> bbox rotates)
2. MD5 dedup prevents duplicates
3. Val/test images must NEVER appear in train (leakage prevention)
4. Background image ratio: max 1.5x of positive images
