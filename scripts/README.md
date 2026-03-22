# Scripts

## Guncel
- `prepare_production_dataset.py` — Uretim veri pipeline'i (60/20/20)
- `train_final_phase1.py` — YOLO egitim + CoordAtt + raporlama
- `export_onnx.py` — ONNX ihracati
- `export_tensorrt.py` — TensorRT ihracati
- `check_gpu.py` — GPU/CUDA kontrol
- `audit_assets.py` — Varlik dogrulama
- `generate_vlm_captions.py` — VLM baslik uretimi (offline)

## Gecmis (veri soyagaci icin saklaniyor)
- `deprecated/train_yolo.py` — Orijinal YOLO egitim (train_final_phase1.py ile degistirildi)
- `deprecated/infer_yolo.py` — Tekli/toplu cikarim (src/pipeline/model_runner.py ile degistirildi)
- `prepare_phase1_dataset.py` — Orijinal Roboflow COCO→YOLO
- `prepare_v2_dataset.py` — Faz 1 v2
- `prepare_v3_copypaste.py` — Copy-paste augmentation
- `prepare_v4_dataset.py` — Faz 1 v4 (tum veriler)
