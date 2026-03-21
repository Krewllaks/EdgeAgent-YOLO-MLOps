# Deployment Guide

## Target Hardware: Jetson Orin Nano 8GB
- GPU: Ampere (1024 CUDA cores)
- RAM: 8GB shared LPDDR5
- Inference budget: ~5ms (1/200fps) TensorRT FP16

## VRAM Budget
| Component | VRAM |
|-----------|------|
| YOLO v10-S + CA | ~500MB |
| PaliGemma NF4 | ~2800MB |
| Overhead | ~400MB |
| **Total** | **~3700MB** |

## TensorRT Export
```bash
python scripts/export_tensorrt.py --half --simplify
# Options: --int8 for INT8 quantization, --workspace 4 (GB)
```

## Edge Profiling
```bash
python src/edge/profiler.py
```
- Measures local GPU latency, projects to Jetson (6.5x multiplier)
- Reports memory budget breakdown

## MQTT Configuration
- Broker: localhost:1883
- Topics:
  - `edgeagent/factory/results` — OK/NOK decisions
  - `edgeagent/factory/vlm_events` — VLM triggers
  - `edgeagent/factory/alerts` — Emergency alerts
- Control topics: emergency_stop, clear_queue, reload_model
- Config: `configs/phase2_config.yaml`

## Latency Profile (RTX 3050)
| Operation | Time |
|-----------|------|
| YOLO inference | ~20-35ms |
| VLM inference (if triggered) | ~3000-4000ms |
| Edge enhancement (optional) | ~50-100ms |
