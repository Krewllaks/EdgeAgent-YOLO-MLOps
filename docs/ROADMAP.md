# Roadmap (Execution)

1. **Data Audit**
   - `python scripts/audit_assets.py`
   - Ciktilar: `reports/generated/data_audit.json`

2. **Phase-1 Dataset Build (Roboflow COCO -> YOLO)**
   - `python scripts/prepare_phase1_dataset.py`
   - Ciktilar: `data/processed/phase1_multiclass_v1/`

3. **YOLO Training Baseline**
   - `python scripts/train_yolo.py --data data/processed/phase1_multiclass_v1/data.yaml --model yolov10s.pt`

4. **Augmented Data Ingestion + Imbalance Analysis**
   - `python src/data/augment_analysis.py`
   - Ciktilar:
     - `reports/generated/augmentation_imbalance_latest.json`
     - `reports/generated/augmentation_imbalance_latest.png`

5. **Final Phase-1 Training (YOLOv10-S + CA)**
   - `python scripts/train_final_phase1.py --data data/processed/phase1_multiclass_v1/data.yaml --batch 8 --imgsz 640 --amp`

6. **Inference + Error Review**
   - `python scripts/infer_yolo.py --model <best.pt> --source <image_or_folder>`

7. **Next Phase**
   - CA module ablation
   - Edge profiler
   - VLM trigger and reasoning
   - Monitoring and CT loop
