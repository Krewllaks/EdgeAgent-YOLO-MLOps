# EdgeAgent-YOLO-MLOps

Scalable MLOps framework for industrial quality control.

This repository contains the Sprint 1 implementation of a two-layer inspection pipeline:

- **Fast detection layer:** YOLOv10-S + Coordinate Attention (CA)
- **Cognitive layer (planned):** PaliGemma-based reasoning for natural-language defect analysis

The current focus is Phase 1: robust OK/NOK detection for `screw`, `missing_screw`, and `missing_component`.

## Sprint 1 Status (Final Phase 1)

- Model: **YOLOv10-S + CA** (`configs/models/yolov10s_ca.yaml`)
- Best validation metric: **mAP50 = 0.9943 (~99.43%)**
- Baseline (plain YOLO smoke run): **mAP50 = 0.49417**
- Improvement: **+0.50004 mAP50**
- Final model artifact: `models/phase1_final_ca.pt`
- Comparison report: `reports/final_phase1_report.md`

## Hardware Requirements

- OS: Windows 10/11
- Python: 3.10+
- GPU: NVIDIA RTX 3050 (4 GB VRAM target)
- CUDA: 12.1 (for GPU training)

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If CUDA is not detected after install, run the explicit CUDA 12.1 wheel command:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

Verify GPU visibility:

```bash
python scripts/check_gpu.py
```

## Workflow (Recommended Order)

### 1) Prepare Dataset

Convert labeled Roboflow export into canonical YOLO split:

```bash
python scripts/prepare_phase1_dataset.py
```

### 2) Augment Integration + Imbalance Analysis

Ingest `coklanmis/` and `coklanmisacili/` data into `train` with duplicate and leakage checks:

```bash
python src/data/augment_analysis.py
python src/data/augment_analysis.py --clean-augmented  # remove old weak labels first
python src/data/augment_analysis.py --source-dir coklanmis --source-dir coklanmisacili
```

Outputs:

- `reports/generated/augmentation_imbalance_latest.json`
- `reports/generated/augmentation_imbalance_latest.png`

### 2b) Edge Enhancement (Canny Preprocessing)

Apply Canny edge blending for metal surface reflection suppression:

```bash
python src/data/edge_enhancer.py --image path/to/img.jpg --preview
python src/data/edge_enhancer.py --input-dir data/processed/phase1_v2/train/images --output-dir data/processed/phase1_v2_edge/train/images
```

### 3) UI Technical Dashboard

Launch the Sprint 1 technical monitoring dashboard:

```bash
streamlit run src/ui/sprint1_dashboard.py
```

Dashboard includes:

- Live inference playground (image upload + YOLO prediction)
- Edge Enhancement preview (Canny blending with parameter tuning)
- Spatial Clustering analysis (geometric post-processing with decision matrix)
- Class balance before/after visualization
- Coordinate Attention rationale
- MLflow experiment tracking integration
- Phase 2 VLM async trigger strategy (visual flow diagram)
- Edge profiler results (Jetson Orin Nano estimate)
- FP analysis and Active Learning feedback
- Decision snapshot and operator controls

### 4) Edge Profiler (Phase 2 Prep)

Benchmark model latency and estimate Jetson Orin Nano performance:

```bash
python src/edge/profiler.py --model models/phase1_final_ca.pt --source data/processed/phase1_multiclass_v1/test/images
```

### 5) VLM Trigger Test (Phase 2 Prep)

Simulate PaliGemma activation on low-confidence detections:

```bash
python src/edge/vlm_trigger.py --model models/phase1_final_ca.pt --source data/processed/phase1_multiclass_v1/test/images --conf-threshold 0.40
```

### 6) TensorRT Export (Edge Deployment)

Export model to TensorRT FP16 engine for Jetson Orin Nano:

```bash
python scripts/export_tensorrt.py --half
python scripts/export_tensorrt.py --dry-run  # config only
```

## Training

### Model V1 (Sprint 1 - Phase 1)

```bash
python scripts/train_final_phase1.py --data data/processed/phase1_multiclass_v1/data.yaml --model-cfg configs/models/yolov10s_ca.yaml --epochs 100 --batch 8 --imgsz 640 --amp --workers 4 --device 0
```

### Model V2 (All data sources)

```bash
python scripts/train_final_phase1.py --data data/processed/phase1_v2/data.yaml --epochs 100 --batch 8 --imgsz 640 --amp --workers 4 --device 0
```

Post-training automation:

- copies best checkpoint to `models/phase1_final_ca.pt` (V1) or `models/phase1_v2_ca.pt` (V2)
- writes comparison report to `reports/`

## Repository Layout

- `src/models/` - shared modules (CoordAtt attention layer)
- `src/edge/` - edge deployment tools (profiler, VLM trigger)
- `src/reasoning/` - spatial logic and geometric clustering post-processor
- `src/ui/` - Streamlit dashboard (11 pages)
- `src/data/` - data processing, augmentation analysis, edge enhancement
- `scripts/` - operational scripts (prepare, train, infer, gpu check, TensorRT export)
- `configs/` - model and dataset configs
- `docs/` - strategy, collaboration guide, setup guides, status
- `reports/` - reproducible reports (`generated/` ignored in git)

## Collaboration Notes

Large/local data and training artifacts are intentionally excluded from git:

- `erdogan1/`, `erdogan2/`, `coklanmis/`, `coklanmisacili/`, `roboflowetiketlenen/`
- `data/processed/`, `runs/`, `mlruns/`, `*.pt`

This keeps the repository lightweight and reproducible for team onboarding.
