# Current Status (Final Phase 1)

Date: 2026-03-10

## Summary

Sprint 1 is complete. Final Phase 1 training finished with YOLOv10-S + Coordinate Attention (CA), and the final model/report artifacts are generated.

## Final Metrics

- Baseline mAP50(B): `0.494170`
- Final mAP50(B): `0.994210`
- Delta: `+0.500040`
- Best observed mAP50(B) during run: `0.994300` (epoch 96)

Sources:

- `runs/detect/runs/smoke/smoke_phase12/results.csv`
- `runs/detect/runs/phase1/yolov10s_ca_final/results.csv`
- `reports/final_phase1_report.md`

## Final Artifacts

- Final best model: `models/phase1_final_ca.pt`
- Training run directory: `runs/detect/runs/phase1/yolov10s_ca_final`
- Final comparison report: `reports/final_phase1_report.md`
- Augmentation analysis report (json): `reports/generated/augmentation_imbalance_latest.json`
- Augmentation analysis chart (png): `reports/generated/augmentation_imbalance_latest.png`

## Dataset Snapshot Used in Sprint 1

- Train images: `4174`
- Train bbox instances:
  - `screw`: `2561`
  - `missing_screw`: `416`
  - `missing_component`: `203`
- Background train images: `2059`

## Environment Notes

- Target training environment: Windows + CUDA 12.1 + RTX 3050 (4 GB)
- GPU verification helper: `scripts/check_gpu.py`
- Setup guide: `docs/GPU_SETUP_WINDOWS.md`

## Phase 2 Preparation (In Progress)

New scripts and documentation added:

- `src/models/coordatt.py` - Shared CoordAtt module (single source of truth)
- `src/edge/profiler.py` - Jetson Orin Nano latency estimator
- `src/edge/vlm_trigger.py` - PaliGemma async trigger logic
- `docs/STRATEGY.md` - Technical strategy (competitive diff, leakage, async VLM)
- `docs/COLLABORATION_GUIDE.md` - Team onboarding and MLflow setup

Key decisions documented:
- TensorRT FP16/INT8 optimization required for 200 products/s target
- VLM trigger threshold: confidence < 0.40
- No new class for "crooked screw" (VLM handles edge cases)
- Dashboard-based operator notification (not SMS/email)
- Weekly continuous training planned

## Next Step (Phase 2)

1. Integrate actual PaliGemma 3B model into VLM worker
2. TensorRT export and Jetson Orin Nano benchmark
3. Continuous training pipeline (weekly auto fine-tune)
4. Concept drift testing (lighting variation)
5. Operator feedback loop (dashboard marking)
