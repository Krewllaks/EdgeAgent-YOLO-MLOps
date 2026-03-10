# Data Sources Used

## 1) Erdogan 1
- Path: `C:/Users/bahti/Desktop/Goruntuisleme/erdogan1`
- Content:
  - `model/` (single-class YOLO export, `vida-ok`)
  - `NOK fotoğraflar/` (etiketsiz NOK ham goruntuler)

## 2) Erdogan 2
- Path: `C:/Users/bahti/Desktop/Goruntuisleme/erdogan2/MLOps Çalışmaları`
- Content:
  - `NOK fotoğraflar/`
  - `OK fotoğraflar/`
  - `models/*.pt`
  - `Codes/*.py`

## 3) Roboflow labeled export (current active source)
- Path: `C:/Users/bahti/Desktop/Goruntuisleme/roboflowetiketlenen`
- Format: COCO (`train/_annotations.coco.json`)
- Images: 897
- Categories observed:
  - `screw`
  - `missing_screw`
  - `missing_component`

## 4) Augmented external set (Sprint-1 boost)
- Path: `C:/Users/bahti/Desktop/Goruntuisleme/coklanmis`
- Folder mapping:
  - `aparatsiz/` -> `missing_component` (ID 2)
  - `eksik_vida/` -> `missing_screw` (ID 1)
  - `ok/` (veya `vida/`) -> `screw` (ID 0)
  - `diger/` -> background (empty label)
- Integrated by: `src/data/augment_analysis.py`
- Leakage policy: val/test hash collision varsa train'e alinmaz
- Duplicate policy: mevcut train hash'iyle ayniysa atlanir

## Canonical dataset output
- `data/processed/phase1_multiclass_v1/`
- Format: YOLO (train/val/test + data.yaml)
