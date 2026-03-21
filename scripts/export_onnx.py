"""
ONNX Export — Platform-bagimsiz model export.

PyTorch .pt modelini ONNX formatina donusturur.
ONNX dosyasi herhangi bir cihazda calisabilir (CPU, CUDA, TensorRT EP).

Kullanim:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model models/phase1_final_ca.pt --half
    python scripts/export_onnx.py --model models/phase1_final_ca.pt --imgsz 640
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="ONNX Export")
    parser.add_argument("--model", type=str,
                        default=str(ROOT / "models" / "phase1_final_ca.pt"),
                        help="PyTorch model yolu")
    parser.add_argument("--imgsz", type=int, default=640, help="Goruntu boyutu")
    parser.add_argument("--half", action="store_true", help="FP16 export")
    parser.add_argument("--simplify", action="store_true", help="ONNX simplify")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    # CoordAtt kayit
    from src.models.coordatt import register_coordatt
    register_coordatt()

    from ultralytics import YOLO

    model = YOLO(args.model)
    print(f"Model: {args.model}")
    print(f"Export: ONNX (imgsz={args.imgsz}, half={args.half}, opset={args.opset})")

    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        simplify=args.simplify,
        opset=args.opset,
    )

    print(f"\n[OK] ONNX dosyasi: {export_path}")

    # Dogrulama
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(export_path))
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        print(f"  Input : {inputs[0].name} {inputs[0].shape}")
        print(f"  Output: {outputs[0].name} {outputs[0].shape}")
        providers = session.get_providers()
        print(f"  Providers: {providers}")
    except ImportError:
        print("  (onnxruntime yuklenmemis, dogrulama atlandi)")


if __name__ == "__main__":
    main()
