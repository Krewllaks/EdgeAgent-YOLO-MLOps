"""Edge Performance Profiler - Jetson Orin Nano latency estimator.

Loads the Phase 1 model, benchmarks inference on the local GPU, then
projects latency and throughput on Jetson Orin Nano using a configurable
scaling factor.

Usage:
    python src/edge/profiler.py \
        --model models/phase1_final_ca.pt \
        --source data/processed/phase1_multiclass_v1/test/images \
        --device 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _register_custom_modules() -> None:
    """Make CoordAtt / HSwish / HSigmoid visible to torch unpickler."""
    sys.path.insert(0, str(ROOT))
    from src.models.coordatt import CoordAtt, HSigmoid, HSwish, register_coordatt

    register_coordatt()

    # Also inject into __main__ so torch.load finds them
    import __main__
    for cls in (HSigmoid, HSwish, CoordAtt):
        setattr(__main__, cls.__name__, cls)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Edge profiler (Jetson Orin Nano estimator)")
    p.add_argument("--model", type=Path, default=ROOT / "models" / "phase1_final_ca.pt")
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "processed" / "phase1_multiclass_v1" / "test" / "images",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--orin-scale",
        type=float,
        default=6.5,
        help="RTX-to-Orin latency multiplier (default: 6.5x for FP16)",
    )
    p.add_argument(
        "--target-fps",
        type=float,
        default=200.0,
        help="Factory target products/sec",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "generated",
    )
    return p.parse_args()


# ── Jetson Orin Nano 8 GB specs (for reference) ─────────────────────
ORIN_SPEC = {
    "gpu": "Ampere (1024 CUDA cores)",
    "tops_sparse": 67,
    "tops_dense": 40,
    "ram_gb": 8,
    "ram_type": "LPDDR5 shared",
    "power_modes_w": [7, 15, 25],
    "tensorrt_note": "INT8/FP16 TensorRT can reduce latency 3-5x vs PyTorch FP32",
}


def main() -> None:
    _register_custom_modules()
    args = parse_args()

    import torch
    from ultralytics import YOLO

    if not args.model.exists():
        sys.exit(f"[ERR] Model not found: {args.model}")
    if not args.source.exists():
        sys.exit(f"[ERR] Source dir not found: {args.source}")

    device = args.device if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.model))

    # Collect image paths
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in args.source.iterdir() if p.suffix.lower() in exts])
    if not images:
        sys.exit(f"[ERR] No images in {args.source}")
    images = images[: args.max_samples]

    # Warmup
    for _ in range(args.warmup):
        model.predict(str(images[0]), imgsz=args.imgsz, device=device, verbose=False)

    # Timed inference
    latencies: list[float] = []
    for img in images:
        t0 = time.perf_counter()
        model.predict(str(img), imgsz=args.imgsz, device=device, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    # Stats
    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)
    p95_ms = sorted(latencies)[int(len(latencies) * 0.95)]

    est_orin_ms = avg_ms * args.orin_scale
    target_budget_ms = 1000.0 / args.target_fps
    needs_trt = est_orin_ms > target_budget_ms

    # Estimate model memory (rough)
    model_size_mb = args.model.stat().st_size / (1024 * 1024)
    est_runtime_mb = model_size_mb * 2.5  # typical runtime overhead
    paligemma_4bit_mb = 2800  # PaliGemma 3B @ 4-bit estimate
    total_est_mb = est_runtime_mb + paligemma_4bit_mb
    orin_fits = total_est_mb < (ORIN_SPEC["ram_gb"] * 1024 * 0.85)  # 85% usable

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": str(args.model),
        "device": device,
        "samples": len(images),
        "imgsz": args.imgsz,
        "local_latency_ms": {
            "avg": round(avg_ms, 3),
            "min": round(min_ms, 3),
            "max": round(max_ms, 3),
            "p95": round(p95_ms, 3),
        },
        "orin_estimate": {
            "scale_factor": args.orin_scale,
            "est_avg_ms": round(est_orin_ms, 3),
            "target_fps": args.target_fps,
            "target_budget_ms": round(target_budget_ms, 3),
            "needs_tensorrt": needs_trt,
        },
        "memory_estimate_mb": {
            "yolo_model_file": round(model_size_mb, 1),
            "yolo_runtime_est": round(est_runtime_mb, 1),
            "paligemma_4bit_est": paligemma_4bit_mb,
            "total_est": round(total_est_mb, 1),
            "orin_ram_mb": ORIN_SPEC["ram_gb"] * 1024,
            "fits_in_orin": orin_fits,
        },
        "orin_spec": ORIN_SPEC,
        "recommendation": (
            "TensorRT FP16/INT8 optimizasyonu GEREKLI. "
            f"PyTorch FP32 tahmini {est_orin_ms:.1f}ms > {target_budget_ms:.1f}ms hedef. "
            "TensorRT ile 3-5x hiz kazanimi beklenir."
            if needs_trt
            else f"PyTorch FP32 yeterli: {est_orin_ms:.1f}ms < {target_budget_ms:.1f}ms hedef."
        ),
    }

    # Save report
    args.output.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output / f"edge_profile_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[OK] Edge profiling complete"
        f" - samples: {len(images)}"
        f" - local avg latency: {avg_ms:.3f} ms"
        f" - est orin latency: {est_orin_ms:.3f} ms"
        f" - target budget: {target_budget_ms:.3f} ms"
        f" - needs TensorRT: {needs_trt}"
        f" - report: {out_path}"
    )


if __name__ == "__main__":
    main()
