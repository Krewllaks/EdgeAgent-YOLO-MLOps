import argparse
import csv
from datetime import datetime
from pathlib import Path
import shutil
from typing import Optional

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


class HSigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3.0) / 6.0


class HSwish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.hsigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.hsigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp: int = 0, reduction: int = 32):
        super().__init__()
        self.inp = int(inp) if inp else None
        self.reduction = max(1, int(reduction))

        self.conv1 = None
        self.bn1 = None
        self.act = HSwish()
        self.conv_h = None
        self.conv_w = None

    def _build(self, channels: int, device: torch.device, dtype: torch.dtype) -> None:
        mip = max(8, channels // self.reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0).to(
            device=device, dtype=dtype
        )
        self.bn1 = nn.BatchNorm2d(mip).to(device=device, dtype=dtype)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0).to(
            device=device, dtype=dtype
        )
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0).to(
            device=device, dtype=dtype
        )
        self.inp = channels

    def _ensure_built(self, x) -> None:
        channels = int(x.shape[1])
        if self.conv1 is None or self.inp != channels:
            self._build(channels, x.device, x.dtype)

    def forward(self, x):
        self._ensure_built(x)
        assert self.conv1 is not None
        assert self.bn1 is not None
        assert self.conv_h is not None
        assert self.conv_w is not None

        identity = x
        _, _, h, w = x.size()

        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final Phase-1 training (YOLOv10-S + CoordAtt)")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "processed" / "phase1_multiclass_v1" / "data.yaml",
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
        default=ROOT / "models" / "phase1_final_ca.pt",
        help="Best model destination path",
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
        default=ROOT / "reports" / "final_phase1_report.md",
        help="Markdown report output path",
    )
    parser.add_argument("--dry-run", action="store_true")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg.lower() == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return device_arg
    print("[WARN] CUDA is not available in current Python env. Falling back to CPU.")
    return "cpu"


def register_coordatt_module() -> None:
    import ultralytics.nn.tasks as tasks

    setattr(tasks, "CoordAtt", CoordAtt)


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
