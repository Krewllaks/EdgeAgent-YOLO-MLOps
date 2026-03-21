"""Mevcut egitim sonuclarini MLflow'a kaydet (one-shot backfill).

Kullanim:
    python scripts/backfill_mlflow.py

Bu script bir kez calistirilir. Daha sonraki egitimlerde
train_final_phase1.py otomatik olarak MLflow'a loglar.
"""
import csv
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlflow

DB_PATH = ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("EdgeAgent-YOLO-Training")

# ── Kaydedilecek run'lar ──
RUNS = [
    {
        "name": "Baseline (smoke_phase12)",
        "run_dir": ROOT / "runs" / "detect" / "runs" / "smoke" / "smoke_phase12",
        "tags": {"version": "baseline", "model": "yolov10s"},
    },
    {
        "name": "Sprint1-V1 (YOLOv10s+CA)",
        "run_dir": ROOT / "runs" / "detect" / "runs" / "phase1" / "yolov10s_ca_final",
        "tags": {"version": "v1", "model": "yolov10s_ca"},
    },
    {
        "name": "Sprint1-V2 (YOLOv10s+CA+Aug)",
        "run_dir": ROOT / "runs" / "detect" / "runs" / "phase1" / "yolov10s_ca_v2",
        "tags": {"version": "v2", "model": "yolov10s_ca"},
    },
]


def parse_results_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_final_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    last = rows[-1]
    metrics = {}
    for key in last:
        val = last[key].strip() if isinstance(last[key], str) else last[key]
        if val == "" or val is None:
            continue
        try:
            metrics[key.strip()] = float(val)
        except (ValueError, TypeError):
            pass
    return metrics


def backfill_run(run_info: dict) -> None:
    name = run_info["name"]
    run_dir = run_info["run_dir"]
    tags = run_info.get("tags", {})

    if not run_dir.exists():
        print(f"[SKIP] {name}: {run_dir} bulunamadi")
        return

    results_csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"

    print(f"\n[LOG] {name}")
    print(f"  dir: {run_dir}")

    with mlflow.start_run(run_name=name, tags=tags):
        # Log hyperparameters from args.yaml
        if args_yaml.exists():
            with args_yaml.open("r", encoding="utf-8") as f:
                args = yaml.safe_load(f)
            important_params = {
                "epochs": args.get("epochs"),
                "batch": args.get("batch"),
                "imgsz": args.get("imgsz"),
                "patience": args.get("patience"),
                "close_mosaic": args.get("close_mosaic"),
                "amp": args.get("amp"),
                "seed": args.get("seed"),
                "optimizer": args.get("optimizer"),
                "lr0": args.get("lr0"),
                "lrf": args.get("lrf"),
                "momentum": args.get("momentum"),
                "weight_decay": args.get("weight_decay"),
                "model": args.get("model", ""),
                "data": args.get("data", ""),
            }
            mlflow.log_params({k: str(v) for k, v in important_params.items() if v is not None})
            print(f"  params: {len(important_params)} logged")

        # Log per-epoch metrics
        rows = parse_results_csv(results_csv)
        metric_keys = [
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
            "metrics/precision(B)", "metrics/recall(B)",
            "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "val/box_loss", "val/cls_loss", "val/dfl_loss",
        ]
        for epoch_idx, row in enumerate(rows):
            for key in metric_keys:
                val = row.get(key, "").strip() if key in row else ""
                # Some CSV headers have leading spaces
                if not val:
                    for k in row:
                        if k.strip() == key:
                            val = row[k].strip()
                            break
                if val:
                    try:
                        clean_key = key.strip().replace("/", "_").replace("(", "").replace(")", "")
                        mlflow.log_metric(clean_key, float(val), step=epoch_idx)
                    except (ValueError, TypeError):
                        pass
        print(f"  epochs: {len(rows)} logged")

        # Log final summary
        final = get_final_metrics(rows)
        for key in ["metrics/mAP50(B)", "metrics/mAP50-95(B)",
                     "metrics/precision(B)", "metrics/recall(B)"]:
            clean = key.strip()
            if clean in final:
                mlflow.log_metric(f"final_{clean.replace('/', '_').replace('(', '').replace(')', '')}", final[clean])
            # Try with leading space
            for k, v in final.items():
                if k.strip() == clean:
                    safe_name = clean.replace("/", "_").replace("(", "").replace(")", "")
                    mlflow.log_metric(f"final_{safe_name}", v)
                    break
        print(f"  final metrics logged")

        # Log plots as artifacts
        plot_count = 0
        for f in run_dir.iterdir():
            if f.suffix in (".png", ".jpg") and f.stat().st_size < 5_000_000:
                mlflow.log_artifact(str(f), artifact_path="plots")
                plot_count += 1
        print(f"  artifacts: {plot_count} plots logged")

        # Log best weights if exists
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="model")
            print(f"  model: best.pt logged")


def main():
    print("=" * 60)
    print("EdgeAgent MLflow Backfill")
    print(f"Tracking URI: sqlite:///{DB_PATH}")
    print("=" * 60)

    for run_info in RUNS:
        backfill_run(run_info)

    print("\n" + "=" * 60)
    print("TAMAMLANDI! MLflow UI'i acmak icin:")
    print(f"  mlflow ui --backend-store-uri sqlite:///{DB_PATH} --port 5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
