"""TensorRT Export Script - Convert YOLO model to TensorRT engine.

Exports the Phase 1 model to TensorRT FP16 (and optionally INT8)
for deployment on Jetson Orin Nano.

Usage:
    python scripts/export_tensorrt.py
    python scripts/export_tensorrt.py --half          # FP16 (default)
    python scripts/export_tensorrt.py --int8           # INT8 quantization
    python scripts/export_tensorrt.py --int8 --half    # INT8 + FP16 fallback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO model to TensorRT engine")
    p.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "phase1_final_ca.pt",
        help="Source .pt model path",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--half", action="store_true", default=True, help="FP16 export (default)")
    p.add_argument("--no-half", dest="half", action="store_false", help="Disable FP16")
    p.add_argument("--int8", action="store_true", help="INT8 quantization")
    p.add_argument("--batch", type=int, default=1, help="Batch size")
    p.add_argument("--workspace", type=int, default=4, help="TensorRT workspace (GB)")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--simplify", action="store_true", default=True, help="ONNX simplify")
    p.add_argument("--dry-run", action="store_true", help="Only show config, do not export")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        sys.exit(f"[ERR] Model not found: {args.model}")

    # Register custom modules
    from src.models.coordatt import CoordAtt, HSigmoid, HSwish, register_coordatt
    register_coordatt()
    import __main__
    for cls in (HSigmoid, HSwish, CoordAtt):
        setattr(__main__, cls.__name__, cls)

    from ultralytics import YOLO

    model = YOLO(str(args.model))

    export_config = {
        "format": "engine",
        "imgsz": args.imgsz,
        "half": args.half,
        "int8": args.int8,
        "batch": args.batch,
        "workspace": args.workspace,
        "device": args.device,
        "simplify": args.simplify,
    }

    print("[INFO] TensorRT Export Configuration:")
    for k, v in export_config.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("[OK] Dry run complete. Export config validated.")
        return

    try:
        engine_path = model.export(**export_config)
        print(f"[OK] TensorRT engine exported: {engine_path}")
        print(f"  - Format: {'FP16' if args.half else 'FP32'}{' + INT8' if args.int8 else ''}")
        print(f"  - Input size: {args.imgsz}x{args.imgsz}")
        print(f"  - Batch size: {args.batch}")
        print(f"\nSonraki adim: Engine dosyasini Jetson Orin Nano'ya kopyalayin:")
        print(f"  scp {engine_path} jetson@<ip>:~/models/")
    except Exception as e:
        print(f"[ERR] Export failed: {e}")
        print("\nNot: TensorRT export icin NVIDIA TensorRT kutuphanesi gereklidir.")
        print("Jetson uzerinde export yapmak daha guvenilirdir.")
        sys.exit(1)


if __name__ == "__main__":
    main()
