"""
Label validator — Weak label detection and unlabeled data control.

Hocanın direktifi: Muhammet'ten gelen çoklanmış veriler label'lı olmalı.
Label'sız veya weak label (fallback bbox "0.5 0.5 0.8 0.8") ile gelen
veriler eğitime sokulmamalı (V2'deki performans düşüşünün sebebi bu).

Bu modül:
1. Veri setindeki weak label'ları tespit eder
2. Label'sız görüntüleri bulur
3. Sınıf bazlı label kalitesi raporu üretir
4. Eğitim öncesi doğrulama (guard) görevi görür
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CLASS_NAMES = {0: "screw", 1: "missing_screw", 2: "missing_component"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class LabelStats:
    total_images: int = 0
    labeled_images: int = 0
    unlabeled_images: int = 0
    weak_labeled_images: int = 0
    strong_labeled_images: int = 0
    empty_label_files: int = 0
    total_boxes: int = 0
    weak_boxes: int = 0
    strong_boxes: int = 0
    per_class_total: dict = field(default_factory=dict)
    per_class_weak: dict = field(default_factory=dict)
    per_class_strong: dict = field(default_factory=dict)
    unlabeled_files: list = field(default_factory=list)
    weak_labeled_files: list = field(default_factory=list)


def is_weak_bbox(cx: float, cy: float, w: float, h: float, threshold: float = 0.70) -> bool:
    """Check if a bounding box is a weak/fallback label.

    Weak labels are typically full-image or near-full-image boxes
    like "0.5 0.5 0.8 0.8" that don't represent actual object locations.
    """
    return w > threshold and h > threshold


def parse_yolo_label(label_path: Path) -> list[dict]:
    """Parse a YOLO format label file."""
    boxes = []
    if not label_path.exists():
        return boxes

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes

    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append({
                "class_id": class_id,
                "cx": cx, "cy": cy, "w": w, "h": h,
                "is_weak": is_weak_bbox(cx, cy, w, h),
            })
        except (ValueError, IndexError):
            continue
    return boxes


def validate_directory(
    image_dir: Path,
    label_dir: Path,
    weak_threshold: float = 0.70,
) -> LabelStats:
    """Validate all labels in a directory pair (images/ + labels/)."""
    stats = LabelStats()

    if not image_dir.exists():
        print(f"[WARN] Image directory not found: {image_dir}")
        return stats

    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    stats.total_images = len(image_files)

    for cls_name in CLASS_NAMES.values():
        stats.per_class_total[cls_name] = 0
        stats.per_class_weak[cls_name] = 0
        stats.per_class_strong[cls_name] = 0

    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            stats.unlabeled_images += 1
            stats.unlabeled_files.append(str(img_path.name))
            continue

        boxes = parse_yolo_label(label_path)

        if not boxes:
            stats.empty_label_files += 1
            stats.labeled_images += 1  # empty = background image (intentional)
            continue

        stats.labeled_images += 1
        has_weak = False

        for box in boxes:
            stats.total_boxes += 1
            cls_name = CLASS_NAMES.get(box["class_id"], f"unknown_{box['class_id']}")

            if cls_name not in stats.per_class_total:
                stats.per_class_total[cls_name] = 0
                stats.per_class_weak[cls_name] = 0
                stats.per_class_strong[cls_name] = 0

            stats.per_class_total[cls_name] += 1

            if box["is_weak"]:
                stats.weak_boxes += 1
                stats.per_class_weak[cls_name] += 1
                has_weak = True
            else:
                stats.strong_boxes += 1
                stats.per_class_strong[cls_name] += 1

        if has_weak:
            stats.weak_labeled_images += 1
            stats.weak_labeled_files.append(str(img_path.name))
        else:
            stats.strong_labeled_images += 1

    return stats


def validate_dataset(dataset_dir: Path, weak_threshold: float = 0.70) -> dict:
    """Validate all splits (train/val/test) of a YOLO dataset."""
    results = {}
    for split in ["train", "val", "test"]:
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        if image_dir.exists():
            results[split] = validate_directory(image_dir, label_dir, weak_threshold)
    return results


def print_report(results: dict, dataset_name: str = "") -> None:
    """Print a console report of label validation results."""
    print("\n" + "=" * 70)
    print(f"  Label Validation Report{f' — {dataset_name}' if dataset_name else ''}")
    print("=" * 70)

    for split, stats in results.items():
        print(f"\n  [{split.upper()}] ({stats.total_images} görüntü)")
        print(f"  {'-' * 50}")
        print(f"  Etiketli       : {stats.labeled_images}")
        print(f"  Etiketsiz      : {stats.unlabeled_images}")
        print(f"  Weak Label'lı  : {stats.weak_labeled_images}")
        print(f"  Strong Label'lı: {stats.strong_labeled_images}")
        print(f"  Boş Label      : {stats.empty_label_files}")
        print()
        print(f"  Toplam Kutu    : {stats.total_boxes}")
        print(f"  Weak Kutu      : {stats.weak_boxes} ({stats.weak_boxes/max(stats.total_boxes,1):.1%})")
        print(f"  Strong Kutu    : {stats.strong_boxes} ({stats.strong_boxes/max(stats.total_boxes,1):.1%})")
        print()

        if stats.per_class_total:
            print(f"  {'Sınıf':<22} {'Toplam':>8} {'Weak':>8} {'Strong':>8} {'Weak%':>8}")
            print(f"  {'-' * 54}")
            for cls_name in CLASS_NAMES.values():
                total = stats.per_class_total.get(cls_name, 0)
                weak = stats.per_class_weak.get(cls_name, 0)
                strong = stats.per_class_strong.get(cls_name, 0)
                pct = f"{weak/total:.1%}" if total > 0 else "N/A"
                print(f"  {cls_name:<22} {total:>8} {weak:>8} {strong:>8} {pct:>8}")

        # Warnings
        if stats.weak_labeled_images > 0:
            print(f"\n  [!] UYARI: {stats.weak_labeled_images} görüntüde weak label tespit edildi!")
            print(f"    Weak label'lar eğitim kalitesini düşürür (V2'deki performans düşüşü buna bağlı).")
            if len(stats.weak_labeled_files) <= 10:
                for f in stats.weak_labeled_files:
                    print(f"      - {f}")
            else:
                for f in stats.weak_labeled_files[:5]:
                    print(f"      - {f}")
                print(f"      ... ve {len(stats.weak_labeled_files) - 5} dosya daha")

        if stats.unlabeled_images > 0:
            print(f"\n  [!] UYARI: {stats.unlabeled_images} görüntüde label dosyası yok!")

    print("\n" + "=" * 70)


def check_training_ready(results: dict) -> bool:
    """Check if dataset is safe to train on (no weak labels in train split).

    Returns True if training is safe, False if weak labels detected.
    """
    train_stats = results.get("train")
    if train_stats is None:
        print("[ERROR] Train split bulunamadı!")
        return False

    if train_stats.weak_labeled_images > 0:
        print(
            f"[BLOCK] Eğitim ENGELLENDI: Train setinde {train_stats.weak_labeled_images} "
            f"weak label'lı görüntü var ({train_stats.weak_boxes} weak bbox)."
        )
        print("  Çözüm: Muhammet'ten label'lı augmented veri isteyin veya weak label'lı verileri çıkarın.")
        return False

    if train_stats.unlabeled_images > 0:
        print(
            f"[WARN] Train setinde {train_stats.unlabeled_images} etiketsiz görüntü var. "
            f"Bunlar background olarak kullanılacak."
        )

    print("[OK] Train seti temiz — weak label yok, eğitime hazır.")
    return True


def save_report(results: dict, output_path: Path) -> None:
    """Save validation report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "splits": {},
    }
    for split, stats in results.items():
        d = asdict(stats)
        # Limit file lists to avoid huge JSON
        d["unlabeled_files"] = d["unlabeled_files"][:50]
        d["weak_labeled_files"] = d["weak_labeled_files"][:50]
        report["splits"][split] = d

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Rapor kaydedildi: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label validator — weak label detection")
    parser.add_argument(
        "--dataset", type=Path,
        default=ROOT / "data" / "processed" / "phase1_multiclass_v1",
        help="Path to YOLO dataset directory (with train/val/test subdirs)",
    )
    parser.add_argument(
        "--weak-threshold", type=float, default=0.70,
        help="Bbox dimension threshold for weak label detection (default: 0.70)",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only check if training is safe (exit code 1 if blocked)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON report path",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"[ERROR] Dataset directory not found: {args.dataset}")
        sys.exit(1)

    results = validate_dataset(args.dataset, args.weak_threshold)

    if args.check_only:
        is_ready = check_training_ready(results)
        sys.exit(0 if is_ready else 1)

    print_report(results, dataset_name=args.dataset.name)
    is_ready = check_training_ready(results)

    if args.output:
        save_report(results, args.output)
    else:
        output_path = ROOT / "reports" / "generated" / f"label_validation_{args.dataset.name}.json"
        save_report(results, output_path)


if __name__ == "__main__":
    main()
