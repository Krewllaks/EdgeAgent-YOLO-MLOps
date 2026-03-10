import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model for Phase-1")
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--model", type=str, default="yolov10s.pt", help="Model checkpoint or yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/phase1")
    parser.add_argument("--name", type=str, default="yolov10s_multiclass")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from e

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    print("[OK] Training finished")


if __name__ == "__main__":
    main()
