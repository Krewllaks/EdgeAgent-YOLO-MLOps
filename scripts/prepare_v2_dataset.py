"""Prepare V2 dataset: Roboflow base + augmented data with folder labels.

Steps:
1. Run prepare_phase1_dataset.py -> data/processed/phase1_v2/
2. Run augment_analysis.py --allow-folder-labels -> adds coklanmis + coklanmisacili

Usage:
    python scripts/prepare_v2_dataset.py
    python scripts/prepare_v2_dataset.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"[STEP] {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[FAIL] {label} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[OK] {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V2 dataset end-to-end")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fallback-bbox",
        type=str,
        default="0.5 0.5 0.8 0.8",
        help="Bbox for folder-based labels (cx cy w h)",
    )
    args = parser.parse_args()

    python = sys.executable
    v2_dir = ROOT / "data" / "processed" / "phase1_v2"

    if args.dry_run:
        print("[DRY RUN] Would execute:")
        print(f"  1. prepare_phase1_dataset.py --output-dir {v2_dir}")
        print(f"  2. augment_analysis.py --dataset-dir {v2_dir} --allow-folder-labels --clean-augmented")
        return

    # Step 1: Create base dataset from Roboflow COCO export
    run(
        [python, str(ROOT / "scripts" / "prepare_phase1_dataset.py"),
         "--output-dir", str(v2_dir)],
        "Create base V2 dataset from Roboflow",
    )

    # Step 2: Ingest augmented data with folder-based labels
    run(
        [python, str(ROOT / "src" / "data" / "augment_analysis.py"),
         "--dataset-dir", str(v2_dir),
         "--allow-folder-labels",
         "--fallback-bbox", args.fallback_bbox,
         "--clean-augmented"],
        "Ingest augmented data (coklanmis + coklanmisacili)",
    )

    # Verify data.yaml exists
    data_yaml = v2_dir / "data.yaml"
    if data_yaml.exists():
        print(f"\n[OK] V2 dataset ready at: {v2_dir}")
        print(f"[OK] data.yaml: {data_yaml}")
        print("\nTo train Model V2:")
        print(f"  python scripts/train_final_phase1.py --data {data_yaml} --epochs 100 --batch 8 --imgsz 640 --amp --workers 4 --device 0")
    else:
        print(f"[WARN] data.yaml not found at {data_yaml}")


if __name__ == "__main__":
    main()
