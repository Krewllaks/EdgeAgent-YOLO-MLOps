# EdgeAgent System Overview

## Architecture
Two-layer hybrid quality control system for industrial production lines.

### Layer 1 — Speed Layer (Real-time)
- **Model:** YOLOv10-S + Coordinate Attention
- **Purpose:** Sub-10ms OK/NOK detection at 200 items/sec
- **Classes:** screw (0), missing_screw (1), missing_component (2)
- **Config:** `configs/models/yolov10s_ca.yaml`

### Layer 2 — Cognitive Layer (Async)
- **Model:** PaliGemma 3B VLM (google/paligemma-3b-mix-224)
- **Purpose:** Deep analysis when YOLO confidence < 0.40
- **Pattern:** Warm Standby (stays in VRAM), Priority Queue
- **Latency:** ~3-4 seconds per query

### Decision Flow
1. YOLO detects objects
2. Spatial Logic validates geometry (K-means k=4, left/right sides)
3. If YOLO and Spatial agree -> Consensus (fast path, skip VLM)
4. If conflict -> VLM arbitrates (VLM-as-Judge)
5. If VLM fails -> Spatial Logic decision (Fail-Safe)

### Key Files
| Module | File | Purpose |
|--------|------|---------|
| CoordAtt | `src/models/coordatt.py` | Attention mechanism |
| VLM | `src/reasoning/vlm_reasoner.py` | PaliGemma inference |
| Spatial | `src/reasoning/spatial_logic.py` | Geometric validation |
| Conflict | `src/reasoning/conflict_resolver.py` | Decision arbitration |
| RCA | `src/reasoning/rca_templates.py` | Turkish root cause analysis |
| Rules | `src/reasoning/dynamic_rules.py` | Config-driven rules |
| Edge | `src/data/edge_enhancer.py` | Canny edge enhancement |
| Trigger | `src/edge/vlm_trigger.py` | Async VLM queue |
| MQTT | `src/edge/mqtt_bridge.py` | IoT communication |
| Profiler | `src/edge/profiler.py` | Jetson latency estimation |
| Dashboard | `src/ui/sprint1_dashboard.py` | 9-page Streamlit UI |
| Agent | `src/agent/orchestrator.py` | Chat-based pipeline control |
