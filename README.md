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

Ingest `coklanmis/` data into `train` with duplicate and leakage checks, then generate reports:

```bash
python src/data/augment_analysis.py
```

Outputs:

- `reports/generated/augmentation_imbalance_latest.json`
- `reports/generated/augmentation_imbalance_latest.png`

### 3) UI Technical Dashboard

Launch the Sprint 1 technical monitoring dashboard:

```bash
streamlit run src/ui/sprint1_dashboard.py
```

Dashboard includes:

- class balance before/after visualization
- Neden CA kullandik? (Coordinate Attention rationale)
- MLflow experiment tracking integration
- Phase 2 VLM async trigger strategy (visual flow diagram)
- Edge profiler results (Jetson Orin Nano estimate)
- Sprint 1 decision snapshot
- Operator controls (emergency stop placeholder)

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

## Phase 2 Preparation Utilities

### Edge profiler (Jetson Orin Nano simulation)

```bash
python src/edge/profiler.py --model models/phase1_final_ca.pt --source data/processed/phase1_multiclass_v1/test/images
```

Output:

- `reports/generated/edge_profile_latest.json`

### Async VLM trigger (YOLO confidence based)

```bash
python src/edge/vlm_trigger.py --model models/phase1_final_ca.pt --source data/processed/phase1_multiclass_v1/test/images --conf-threshold 0.40
```

Outputs:

- `reports/generated/vlm_trigger_events.jsonl`
- `reports/generated/vlm_trigger_events.summary.json`

## Train Final Phase 1 (Optional Re-run)

```bash
python scripts/train_final_phase1.py --data data/processed/phase1_multiclass_v1/data.yaml --model-cfg configs/models/yolov10s_ca.yaml --epochs 100 --batch 8 --imgsz 640 --amp --workers 4 --device 0
```

Post-training automation:

- copies best checkpoint to `models/phase1_final_ca.pt`
- writes baseline-vs-final comparison to `reports/final_phase1_report.md`

## Repository Layout

- `src/models/` - shared modules (CoordAtt attention layer)
- `src/edge/` - edge deployment tools (profiler, VLM trigger)
- `src/ui/` - Streamlit dashboard
- `src/data/` - data processing and augmentation analysis
- `scripts/` - operational scripts (prepare, train, infer, gpu check)
- `configs/` - model and dataset configs
- `docs/` - strategy, collaboration guide, setup guides, status
- `reports/` - reproducible reports (`generated/` ignored in git)

## Collaboration Notes

Large/local data and training artifacts are intentionally excluded from git:

- `erdogan1/`, `erdogan2/`, `coklanmis/`, `roboflowetiketlenen/`
- `data/processed/`, `runs/`, `mlruns/`, `*.pt`

This keeps the repository lightweight and reproducible for team onboarding.

## Additional Docs

- `docs/STATUS_CURRENT.md` - final Sprint 1 status and artifact paths
- `docs/STRATEGY.md` - technical decisions and Phase 2 trigger strategy
- `docs/COLLABORATION_GUIDE.md` - teammate setup + MLflow usage
- `docs/GPU_SETUP_WINDOWS.md` - CUDA 12.1 setup details
