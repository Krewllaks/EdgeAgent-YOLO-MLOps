# Final Phase 1 Report

- Timestamp: 2026-03-10T09:45:32
- Model cfg: `configs\models\yolov10s_ca.yaml`
- Data yaml: `data\processed\phase1_multiclass_v1\data.yaml`
- Pretrained: `yolov10s.pt`
- Hyperparams: epochs=100, batch=8, imgsz=640, amp=True, workers=4
- Run dir: `C:\Users\bahti\Desktop\Goruntuisleme\runs\detect\runs\phase1\yolov10s_ca_final`
- Best model copied to: `C:\Users\bahti\Desktop\Goruntuisleme\models\phase1_final_ca.pt`

## Baseline vs Final

- Baseline source: `C:\Users\bahti\Desktop\Goruntuisleme\runs\detect\runs\smoke\smoke_phase12\results.csv`
- Final results source: `C:\Users\bahti\Desktop\Goruntuisleme\runs\detect\runs\phase1\yolov10s_ca_final\results.csv`

| Metric | Baseline | Final | Delta |
|---|---:|---:|---:|
| mAP50(B) | 0.494170 | 0.994210 | 0.500040 |
