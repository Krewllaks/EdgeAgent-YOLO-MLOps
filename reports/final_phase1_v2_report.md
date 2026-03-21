# Final Phase 1 Report

- Timestamp: 2026-03-12T02:45:31
- Model cfg: `configs/models/yolov10s_ca.yaml`
- Data yaml: `data/processed/phase1_v2/data.yaml`
- Pretrained: `yolov10s.pt`
- Hyperparams: epochs=100, batch=8, imgsz=640, amp=True, workers=4
- Run dir: `runs/detect/runs/phase1/yolov10s_ca_v2`
- Best model copied to: `models/phase1_v2_ca.pt`

## Baseline vs Final

- Baseline source: `runs/detect/runs/smoke/smoke_phase12/results.csv`
- Final results source: `runs/detect/runs/phase1/yolov10s_ca_v2/results.csv`

| Metric | Baseline | Final | Delta |
|---|---:|---:|---:|
| mAP50(B) | 0.494170 | 0.844950 | 0.350780 |
