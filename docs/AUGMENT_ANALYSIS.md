# Augmented Data Integration Logic

Script: `src/data/augment_analysis.py`

## Input

- Source root: `coklanmis/`
  - `aparatsiz/` -> class 2 (`missing_component`)
  - `eksik_vida/` -> class 1 (`missing_screw`)
  - `ok/` or `vida/` -> class 0 (`screw`)
  - `diger/` -> background (empty label)
- Target dataset: `data/processed/phase1_multiclass_v1/`

## What the script does

1. Reads current train distribution from `train/labels` (before snapshot).
2. Builds hash index of existing images:
   - train hashes for duplicate prevention
   - val/test hashes for leakage prevention
3. Ingests source images into `train/images` + `train/labels`:
   - duplicate hash -> skip
   - val/test hash collision -> skip
   - class folders -> weak box label (`0.5,0.5,0.98,0.98`)
   - background folder -> empty label file
4. Applies background cap using:
   - `max_background`
   - `background_ratio * positive_candidates`
5. Writes reports:
   - `reports/generated/augmentation_imbalance_latest.json`
   - `reports/generated/augmentation_imbalance_latest.png`

## Command

```bash
python src/data/augment_analysis.py
```

Optional:

```bash
python src/data/augment_analysis.py --max-background 1500 --background-ratio 1.2 --seed 42
```
