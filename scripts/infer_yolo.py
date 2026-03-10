import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO inference on folder")
    parser.add_argument("--model", type=Path, required=True, help="Path to .pt model")
    parser.add_argument("--source", type=Path, required=True, help="Image file or folder")
    parser.add_argument("--project", type=str, default="runs/infer")
    parser.add_argument("--name", type=str, default="phase1_demo")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.source.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from e

    model = YOLO(str(args.model))
    model.predict(
        source=str(args.source),
        conf=args.conf,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=True,
        save_conf=True,
    )
    print("[OK] Inference finished")


if __name__ == "__main__":
    main()
