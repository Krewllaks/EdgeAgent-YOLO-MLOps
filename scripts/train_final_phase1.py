import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
import shutil
import sys
from typing import Optional

import torch

try:
    import mlflow
    MLFLOW_OK = True
except ImportError:
    MLFLOW_OK = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.coordatt import CoordAtt, HSigmoid, HSwish, register_coordatt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final Phase-1 training (YOLOv10-S + CoordAtt)")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "processed" / "phase1_multiclass_v1" / "data.yaml",
        help="Path to data.yaml (use phase1_v2 for Model V2)",
    )
    parser.add_argument(
        "--model-cfg",
        type=Path,
        default=ROOT / "configs" / "models" / "yolov10s_ca.yaml",
    )
    parser.add_argument("--pretrained", type=str, default="yolov10s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/phase1")
    parser.add_argument("--name", type=str, default="yolov10s_ca_final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-best",
        type=Path,
        default=None,
        help="Best model destination path (auto-detected from data version)",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=ROOT / "runs" / "detect" / "runs" / "smoke" / "smoke_phase12" / "results.csv",
        help="Baseline YOLO results.csv path for comparison",
    )
    parser.add_argument(
        "--final-report",
        type=Path,
        default=None,
        help="Markdown report output path (auto-detected from data version)",
    )
    parser.add_argument("--dry-run", action="store_true")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    args = parser.parse_args()

    # Auto-detect output paths from data version
    data_dir_name = args.data.parent.name  # e.g. "phase1_multiclass_v1" or "phase1_v2"
    is_v2 = "v2" in data_dir_name

    if args.output_best is None:
        if is_v2:
            args.output_best = ROOT / "models" / "phase1_v2_ca.pt"
        else:
            args.output_best = ROOT / "models" / "phase1_final_ca.pt"

    if args.final_report is None:
        if is_v2:
            args.final_report = ROOT / "reports" / "final_phase1_v2_report.md"
        else:
            args.final_report = ROOT / "reports" / "final_phase1_report.md"

    if is_v2 and args.name == "yolov10s_ca_final":
        args.name = "yolov10s_ca_v2"

    return args


def resolve_device(device_arg: str) -> str:
    if device_arg.lower() == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return device_arg
    print("[WARN] CUDA is not available in current Python env. Falling back to CPU.")
    return "cpu"


def register_coordatt_module() -> None:
    register_coordatt()
    # Also register in __main__ for backward-compatible checkpoint loading
    import __main__
    for cls in (HSigmoid, HSwish, CoordAtt):
        setattr(__main__, cls.__name__, cls)


def parse_map50_from_csv(results_csv: Path) -> Optional[float]:
    if not results_csv.exists():
        return None

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row]

    if not rows:
        return None

    for row in reversed(rows):
        value = row.get("metrics/mAP50(B)")
        if value is None or value == "":
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _log_epoch_metrics(results_csv: Path) -> None:
    """Log per-epoch metrics from YOLO results.csv to MLflow."""
    if not MLFLOW_OK or not results_csv.exists():
        return
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    metric_keys = [
        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "metrics/precision(B)", "metrics/recall(B)",
        "train/box_loss", "train/cls_loss", "train/dfl_loss",
        "val/box_loss", "val/cls_loss", "val/dfl_loss",
    ]
    for epoch_idx, row in enumerate(rows):
        for key in metric_keys:
            val = row.get(key, "").strip()
            if val:
                try:
                    mlflow_key = key.replace("/", "_").replace("(", "").replace(")", "")
                    mlflow.log_metric(mlflow_key, float(val), step=epoch_idx)
                except (ValueError, TypeError):
                    pass


def resolve_run_dir(model, explicit_project: str, explicit_name: str) -> Path:
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        return Path(save_dir)

    project_dir = ROOT / explicit_project
    if not project_dir.exists():
        return project_dir / explicit_name

    candidates = sorted(
        [p for p in project_dir.iterdir() if p.is_dir() and p.name.startswith(explicit_name)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return project_dir / explicit_name


def write_final_report(
    report_path: Path,
    args: argparse.Namespace,
    run_dir: Path,
    final_map50: Optional[float],
    baseline_map50: Optional[float],
    best_dst: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    delta = None
    if final_map50 is not None and baseline_map50 is not None:
        delta = final_map50 - baseline_map50

    def fmt(v: Optional[float]) -> str:
        return "N/A" if v is None else f"{v:.6f}"

    lines = [
        "# Final Phase 1 Report",
        "",
        f"- Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"- Model cfg: `{args.model_cfg}`",
        f"- Data yaml: `{args.data}`",
        f"- Pretrained: `{args.pretrained}`",
        f"- Hyperparams: epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, amp={args.amp}, workers={args.workers}",
        f"- Run dir: `{run_dir}`",
        f"- Best model copied to: `{best_dst}`",
        "",
        "## Baseline vs Final",
        "",
        f"- Baseline source: `{args.baseline_results}`",
        f"- Final results source: `{run_dir / 'results.csv'}`",
        "",
        "| Metric | Baseline | Final | Delta |",
        "|---|---:|---:|---:|",
        f"| mAP50(B) | {fmt(baseline_map50)} | {fmt(final_map50)} | {fmt(delta)} |",
        "",
    ]

    if baseline_map50 is None:
        lines.append("- Not: Baseline mAP50 bulunamadi. `--baseline-results` yolunu dogrulayin.")
    if final_map50 is None:
        lines.append("- Not: Final mAP50 okunamadi. Egitim ciktisindaki `results.csv` dosyasini kontrol edin.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")
    if not args.model_cfg.exists():
        raise FileNotFoundError(f"model cfg not found: {args.model_cfg}")

    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError("Ultralytics is required. Run: pip install -r requirements.txt") from e

    register_coordatt_module()
    device = resolve_device(args.device)

    model = YOLO(str(args.model_cfg))
    if args.pretrained:
        model = model.load(args.pretrained)

    if args.dry_run:
        print("[OK] Dry run complete. Model initialized with CoordAtt.")
        print(f"- model cfg: {args.model_cfg}")
        print(f"- pretrained: {args.pretrained}")
        print(f"- data: {args.data}")
        print(f"- device: {device}")
        print(f"- epochs(default): {args.epochs}")
        print(f"- batch(default): {args.batch}")
        print(f"- imgsz(default): {args.imgsz}")
        print(f"- workers(default): {args.workers}")
        print(f"- amp(default): {args.amp}")
        return

    # ── MLflow setup ──
    if MLFLOW_OK:
        # Use MLflow server if running, otherwise fallback to local SQLite
        mlflow_server = os.environ.get("MLFLOW_TRACKING_URI", "")
        if mlflow_server:
            mlflow.set_tracking_uri(mlflow_server)
        else:
            mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        mlflow.set_experiment("EdgeAgent-YOLO-Training")
        mlflow.start_run(run_name=args.name)
        mlflow.log_params({
            "model_cfg": str(args.model_cfg.name),
            "pretrained": args.pretrained,
            "data": str(args.data.name),
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "amp": args.amp,
            "workers": args.workers,
            "seed": args.seed,
            "device": device,
            "close_mosaic": 10,
            "patience": 25,
        })
        print("[MLflow] Run started")

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        amp=args.amp,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        cache=False,
        close_mosaic=10,
        patience=25,
    )

    run_dir = resolve_run_dir(model, args.project, args.name)
    best_src = run_dir / "weights" / "best.pt"
    args.output_best.parent.mkdir(parents=True, exist_ok=True)
    if best_src.exists():
        shutil.copy2(best_src, args.output_best)
        print(f"[OK] Best weights copied: {args.output_best}")
    else:
        print(f"[WARN] best.pt not found at: {best_src}")

    final_results_csv = run_dir / "results.csv"
    final_map50 = parse_map50_from_csv(final_results_csv)
    baseline_map50 = parse_map50_from_csv(args.baseline_results)

    # ── MLflow: log metrics & artifacts ──
    if MLFLOW_OK:
        # Log epoch-by-epoch metrics from results.csv
        if final_results_csv.exists():
            _log_epoch_metrics(final_results_csv)

        # Log final summary metrics
        if final_map50 is not None:
            mlflow.log_metric("final_mAP50", final_map50)
        if baseline_map50 is not None:
            mlflow.log_metric("baseline_mAP50", baseline_map50)
            if final_map50 is not None:
                mlflow.log_metric("delta_mAP50", final_map50 - baseline_map50)

        # Log best model as artifact
        if best_src.exists():
            mlflow.log_artifact(str(args.output_best), artifact_path="model")

        # Log training curves if available
        for plot_name in ["results.png", "confusion_matrix.png", "PR_curve.png",
                          "F1_curve.png", "P_curve.png", "R_curve.png"]:
            plot_path = run_dir / plot_name
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path), artifact_path="plots")

        mlflow.end_run()
        print("[MLflow] Run logged successfully")

    write_final_report(
        report_path=args.final_report,
        args=args,
        run_dir=run_dir,
        final_map50=final_map50,
        baseline_map50=baseline_map50,
        best_dst=args.output_best,
    )

    print("[OK] Final Phase-1 training finished")
    print(f"- run_dir: {run_dir}")
    print(f"- final report: {args.final_report}")


if __name__ == "__main__":
    main()
