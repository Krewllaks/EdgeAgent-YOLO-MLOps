# Training Guide

## Prerequisites
1. Run `label_validator.py --check-only` to verify no weak labels
2. Ensure CUDA is available: `python scripts/check_gpu.py`
3. Register CoordAtt before loading model: `register_coordatt()`

## Training Command
```bash
python scripts/train_final_phase1.py \
  --data data/processed/phase1_multiclass_v1/data.yaml \
  --model-cfg configs/models/yolov10s_ca.yaml \
  --epochs 100 --batch 8 --imgsz 640 --amp --workers 4 --device 0
```

## Key Parameters
| Parameter | Default | Notes |
|-----------|---------|-------|
| epochs | 100 | Phase 1 achieved best at ~96 |
| batch | 8 | RTX 3050 4GB VRAM limit |
| imgsz | 640 | Standard YOLO input size |
| amp | True | Mixed precision (faster) |
| patience | 25 | Early stopping |
| close_mosaic | 10 | Disable mosaic last 10 epochs |

## Success Metrics
- **Primary:** mAP50 (target: >0.99)
- **Critical:** missing_screw Recall (must catch ALL missing screws)
- **Critical:** missing_component Recall
- False Negative rate must be near zero (safety-critical)

## After Training
1. Best weights auto-copied to `models/phase1_final_ca.pt`
2. Results CSV at `runs/detect/runs/phase1/<name>/results.csv`
3. Run accuracy report: `python src/evaluation/accuracy_report.py --model models/phase1_final_ca.pt`
4. MLflow metrics logged to `mlflow.db`

## V1 vs V2 Performance
| Version | mAP50 | Issue |
|---------|-------|-------|
| V1 | 0.9943 | Clean labels, strong performance |
| V2 | 0.8449 | Weak labels poisoned training |

V2 failure root cause: 1611 augmented images with fallback bbox "0.5 0.5 0.8 0.8"
