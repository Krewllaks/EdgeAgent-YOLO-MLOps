# Troubleshooting Guide

## Common Issues

### "CoordAtt not found" / YAML parse error
**Cause:** `register_coordatt()` not called before model load.
**Fix:** Always call `register_coordatt()` before `YOLO(model_cfg)`.

### VLM OOM (Out of Memory)
**Cause:** PaliGemma + YOLO exceed 4GB VRAM.
**Fix:** System auto-falls back to YOLO-only mode. Or unload VLM first.

### V2 model performs worse than V1
**Cause:** Weak labels (fallback bbox "0.5 0.5 0.8 0.8") in training data.
**Fix:** Run `python src/data/label_validator.py --check-only` before training.
Remove or re-label weak-labeled images.

### MQTT connection refused
**Cause:** No MQTT broker running.
**Fix:** System auto-switches to simulation mode. Or start Mosquitto broker.

### Training stuck at low mAP
**Possible causes:**
1. Class imbalance (screw >> missing_screw >> missing_component)
2. Weak labels in training set
3. Data leakage between train/val/test

**Fix:** Run `label_validator.py` and `augment_analysis.py` diagnostics.

### Spatial Logic gives wrong verdict
**Cause:** Hard-coded rules don't match product geometry.
**Fix:** Update `configs/rules.yaml` or use dynamic rule generation:
```python
from src.reasoning.dynamic_rules import RuleEngine
engine = RuleEngine()
rule = engine.generate_rule_from_spec("6 vidali, sol 3 sag 3")
engine.add_product(rule)
```

### Edge Enhancement makes image worse
**Cause:** Auto-tune Canny parameters not optimal for current lighting.
**Fix:** Run `auto_tune_fast()` with current images. Check domain constraints.

## Error Recovery
| Error | Auto-Recovery |
|-------|---------------|
| VLM OOM | Switch to YOLO-only mode |
| VLM timeout | Spatial Logic fail-safe |
| YOLO impossible output (>4 screws) | Reject + auto-tune |
| SSIM drift >15% | Retrain recommendation + MQTT alert |
| MQTT disconnect | Simulation mode |
