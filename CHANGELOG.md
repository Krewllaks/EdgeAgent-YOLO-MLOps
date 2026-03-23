# Changelog

Tum onemli degisiklikler bu dosyada belgelenir.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [1.1.0] - 2026-03-24

### Added
- Canli kamera akisi (MJPEG) — `/api/stream` endpoint, bbox + verdict overlay
- `src/pipeline/frame_buffer.py` — Thread-safe circular frame buffer
- `src/ui/stream_utils.py` — Frame annotation + MJPEG generator
- HMI'da "Canli Kamera" karti (baslat/durdur toggle)
- `.dockerignore` — Docker image boyutu optimizasyonu
- Multi-stage Dockerfile (builder + production)
- GitHub Actions CI pipeline (lint + test + docker build)
- `CHANGELOG.md`, `.env.example`
- `src/common/constants.py` — Merkezi sabitler (CLASS_NAMES, IMAGE_EXTS)
- `src/common/config.py` — Merkezi YAML config loader
- `src/mlops/vlm_labeler.py` — VLM pseudo-label uretimi (CT pipeline)
- `pyproject.toml` — ruff linter + pytest konfigurasyonu
- `tests/` — Temel pytest suite (spatial_logic, conflict_resolver, rca, config, constants)

### Changed
- Docker image boyutu: ~18GB → ~3-4GB (.dockerignore + multi-stage)
- `deploy/requirements-docker.txt` — Tum versiyonlar pinlendi (reproducible builds)
- `configs/production_config.yaml` — Stream ayarlari eklendi
- `src/pipeline/inference_pipeline.py` — FrameBuffer entegrasyonu, MockCamera fallback
- `src/ui/production_hmi.py` — MJPEG stream endpoint, canli kamera HTML karti
- Silent `except: pass` → `logger.warning/debug` (9 dosya)
- DRY: CLASS_NAMES/IMAGE_EXTS 9 dosyadan `src/common/constants.py`'a tasindi

### Removed
- `src/edgeagent/` bos paket silindi
- Legacy scriptler `scripts/deprecated/`'a tasindi

## [1.0.0] - 2026-03-20

### Added
- YOLOv10-S + Coordinate Attention modeli (mAP50: 0.9943)
- PaliGemma 3B VLM bilissel katman (belirsiz kare analizi)
- Spatial Logic — K-means 4-kumeleme + geometrik validasyon
- Conflict Resolver — YOLO/Spatial/VLM arbitrasyon
- Production HMI (FastAPI) — operator arayuzu
- Streamlit dashboard — 9 sayfa gelistirme araci
- Docker deployment (nvidia/cuda base)
- MQTT bridge, OPC-UA entegrasyonu
- Active Learning pipeline (operator geri bildirimi)
- Concept Drift detector (SSIM)
- Continuous Training pipeline
- Model Registry + Shadow deployment
- Watchdog + Audit Logger
