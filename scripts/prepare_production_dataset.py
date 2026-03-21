"""
Production Dataset Preparation — %60/%20/%20 split + dengeli hata siniflari.

Orijinal fabrika goruntulerini 60/20/20 olarak boler.
Her split icin AYRI base image'lar kullanarak copy-paste augmentation yapar.
Val/test augmented goruntuler FARKLI base'ler kullanir (leakage onleme).

Kullanim:
    python scripts/prepare_production_dataset.py
    python scripts/prepare_production_dataset.py --dry-run
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_v3_copypaste import (
    load_crops,
    find_screw_positions,
    generate_augmented_image,
    md5_file,
    IMAGE_EXTS,
    CLASS_NAMES,
)

# All crop sources
CROP_SOURCES = [
    ROOT / "coklanmis1000",
    ROOT / "coklanmisyeni",
    ROOT / "coklanmis",
    ROOT / "coklanmisacili",
]

# Extra labeled data (screw-only, class 0)
ERDOGAN_YOLO = ROOT / "erdogan1" / "model"


def load_all_crops():
    """Load and deduplicate all missing_screw + missing_component crops."""
    all_ms, all_mc = [], []
    for src in CROP_SOURCES:
        if not src.exists():
            continue
        ms_dir, mc_dir = src / "missing_screw", src / "missing_component"
        if ms_dir.exists():
            all_ms.extend(load_crops(ms_dir, class_id=1))
        if mc_dir.exists():
            all_mc.extend(load_crops(mc_dir, class_id=2))

    # Dedup by image hash
    seen = set()
    unique_ms, unique_mc = [], []
    for crop in all_ms:
        h = hashlib.md5(crop["image"].tobytes()[:2048]).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_ms.append(crop)
    for crop in all_mc:
        h = hashlib.md5(crop["image"].tobytes()[:2048]).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_mc.append(crop)
    return unique_ms, unique_mc


def collect_all_original_images(v1_dir: Path, erdogan_dir: Path):
    """Collect all original (non-augmented) images with labels."""
    images = []  # list of (img_path, lbl_path_or_None, source)

    # V1 images (3-class labels)
    for split in ["train", "val", "test"]:
        img_dir = v1_dir / split / "images"
        lbl_dir = v1_dir / split / "labels"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = lbl_dir / f"{p.stem}.txt"
            images.append((p, lbl if lbl.exists() else None, "v1"))

    # Erdogan images (screw-only labels)
    erd_img = erdogan_dir / "train" / "images"
    erd_lbl = erdogan_dir / "train" / "labels"
    if erd_img.exists():
        for p in sorted(erd_img.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = erd_lbl / f"{p.stem}.txt"
            images.append((p, lbl if lbl.exists() else None, "erdogan"))

    # Dedup by MD5
    seen = set()
    unique = []
    for img_path, lbl_path, source in images:
        h = md5_file(img_path)
        if h not in seen:
            seen.add(h)
            unique.append((img_path, lbl_path, source))

    return unique


def has_defect_label(lbl_path):
    """Check if label file contains missing_screw or missing_component."""
    if lbl_path is None or not lbl_path.exists():
        return False
    text = lbl_path.read_text(encoding="utf-8").strip()
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 5 and int(parts[0]) in (1, 2):
            return True
    return False


def generate_split_augmented(base_images, base_lbl_dir_map, ms_crops, mc_crops, count, rng, prefix):
    """Generate augmented images for a specific split using its own base images."""
    pairs = []
    for i in range(count):
        img_path, lbl_path, _ = rng.choice(base_images)
        bg = cv2.imread(str(img_path))
        if bg is None:
            continue
        bg_labels = find_screw_positions(lbl_path) if lbl_path and lbl_path.exists() else []
        if not bg_labels:
            continue
        aug_img, aug_labels = generate_augmented_image(bg, bg_labels, ms_crops, mc_crops, rng)
        pairs.append({"image": aug_img, "labels": aug_labels, "index": i})
        if (i + 1) % 100 == 0:
            print(f"    {prefix}: {i + 1}/{count}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Production Dataset — 60/20/20 split")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "processed" / "production_v1",
                        help="Output dataset directory")
    parser.add_argument("--train-aug", type=int, default=2000,
                        help="Augmented images for train")
    parser.add_argument("--val-aug", type=int, default=200,
                        help="Augmented images for val (different base)")
    parser.add_argument("--test-aug", type=int, default=200,
                        help="Augmented images for test (different base)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("=" * 60)
    print("  Production Dataset — 60/20/20 Dengeli Split")
    print("=" * 60)

    # ── 1. Collect all original images ──────────────────────────
    print("\n[1/5] Tum orijinal goruntuler toplanıyor...")
    all_images = collect_all_original_images(
        ROOT / "data" / "processed" / "phase1_multiclass_v1",
        ERDOGAN_YOLO,
    )
    print(f"  Toplam unique goruntu: {len(all_images)}")

    # Separate defect vs non-defect for stratified split
    defect_imgs = [(p, l, s) for p, l, s in all_images if has_defect_label(l)]
    normal_imgs = [(p, l, s) for p, l, s in all_images if not has_defect_label(l)]
    print(f"  Hata etiketli: {len(defect_imgs)}")
    print(f"  Normal: {len(normal_imgs)}")

    # ── 2. Stratified split ─────────────────────────────────────
    print("\n[2/5] Stratified 60/20/20 split...")
    rng.shuffle(defect_imgs)
    rng.shuffle(normal_imgs)

    def split_list(items, ratios=(0.6, 0.2, 0.2)):
        n = len(items)
        t1 = int(n * ratios[0])
        t2 = int(n * (ratios[0] + ratios[1]))
        return items[:t1], items[t1:t2], items[t2:]

    # Split defects — ensure each split gets some
    d_train, d_val, d_test = split_list(defect_imgs)
    # Split normals
    n_train, n_val, n_test = split_list(normal_imgs)

    train_imgs = d_train + n_train
    val_imgs = d_val + n_val
    test_imgs = d_test + n_test

    rng.shuffle(train_imgs)
    rng.shuffle(val_imgs)
    rng.shuffle(test_imgs)

    print(f"  Train: {len(train_imgs)} (defect: {len(d_train)})")
    print(f"  Val:   {len(val_imgs)} (defect: {len(d_val)})")
    print(f"  Test:  {len(test_imgs)} (defect: {len(d_test)})")

    # ── 3. Load crops ──────────────────────────────────────────
    print("\n[3/5] Crop'lar yukleniyor...")
    ms_crops, mc_crops = load_all_crops()
    print(f"  missing_screw: {len(ms_crops)}, missing_component: {len(mc_crops)}")

    if args.dry_run:
        print(f"\n[DRY RUN] Ozet:")
        print(f"  Train: {len(train_imgs)} orijinal + {args.train_aug} augmented = {len(train_imgs) + args.train_aug}")
        print(f"  Val:   {len(val_imgs)} orijinal + {args.val_aug} augmented = {len(val_imgs) + args.val_aug}")
        print(f"  Test:  {len(test_imgs)} orijinal + {args.test_aug} augmented = {len(test_imgs) + args.test_aug}")
        print(f"  Output: {args.output}")
        return

    # ── 4. Generate augmented for each split ───────────────────
    print(f"\n[4/5] Augmented goruntuler uretiliyor...")
    print(f"  Train: {args.train_aug} goruntu (train base'lerden)...")
    train_aug = generate_split_augmented(
        train_imgs, None, ms_crops, mc_crops, args.train_aug, rng, "train"
    )
    print(f"  Val: {args.val_aug} goruntu (val base'lerden)...")
    val_aug = generate_split_augmented(
        val_imgs, None, ms_crops, mc_crops, args.val_aug, rng, "val"
    )
    print(f"  Test: {args.test_aug} goruntu (test base'lerden)...")
    test_aug = generate_split_augmented(
        test_imgs, None, ms_crops, mc_crops, args.test_aug, rng, "test"
    )

    # ── 5. Write dataset ───────────────────────────────────────
    print(f"\n[5/5] Dataset yaziliyor...")
    output = args.output
    if output.exists():
        shutil.rmtree(output)

    for split in ["train", "val", "test"]:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy originals
    split_data = [
        ("train", train_imgs, train_aug),
        ("val", val_imgs, val_aug),
        ("test", test_imgs, test_aug),
    ]

    for split_name, orig_imgs, aug_pairs in split_data:
        # Original images
        orig_count = 0
        for img_path, lbl_path, source in orig_imgs:
            dst_img = output / split_name / "images" / img_path.name
            # Handle duplicate filenames from different sources
            if dst_img.exists():
                stem = f"{source}_{img_path.stem}"
                dst_img = output / split_name / "images" / f"{stem}{img_path.suffix}"
            shutil.copy2(img_path, dst_img)
            if lbl_path and lbl_path.exists():
                dst_lbl = output / split_name / "labels" / f"{dst_img.stem}.txt"
                shutil.copy2(lbl_path, dst_lbl)
            orig_count += 1

        # Augmented images
        aug_count = 0
        for pair in aug_pairs:
            fname = f"prod_aug_{split_name}_{pair['index']:04d}"
            dst_img = output / split_name / "images" / f"{fname}.jpg"
            dst_lbl = output / split_name / "labels" / f"{fname}.txt"
            cv2.imwrite(str(dst_img), pair["image"])
            dst_lbl.write_text("\n".join(pair["labels"]), encoding="utf-8")
            aug_count += 1

        print(f"  {split_name}: {orig_count} orijinal + {aug_count} augmented = {orig_count + aug_count}")

    # Class distribution
    print(f"\n  Sinif dagilimi:")
    for split_name in ["train", "val", "test"]:
        counts = {0: 0, 1: 0, 2: 0}
        lbl_dir = output / split_name / "labels"
        for lbl_path in lbl_dir.glob("*.txt"):
            for line in lbl_path.read_text(encoding="utf-8").strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    counts[int(parts[0])] = counts.get(int(parts[0]), 0) + 1
        total = sum(counts.values())
        print(f"    {split_name}: screw={counts[0]}, ms={counts[1]}, mc={counts[2]} (toplam={total})")

    # data.yaml
    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        f"path: {output.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"nc: 3\n"
        f"names:\n"
        f"- screw\n"
        f"- missing_screw\n"
        f"- missing_component\n",
        encoding="utf-8",
    )

    # Summary
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "version": "production_v1",
        "split_ratio": "60/20/20",
        "original_images": len(all_images),
        "defect_images": len(defect_imgs),
        "splits": {
            "train": {"original": len(train_imgs), "augmented": len(train_aug)},
            "val": {"original": len(val_imgs), "augmented": len(val_aug)},
            "test": {"original": len(test_imgs), "augmented": len(test_aug)},
        },
        "crop_sources": [str(s) for s in CROP_SOURCES],
        "seed": args.seed,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"\n  TAMAMLANDI!")
    print(f"  Output: {output}")
    print(f"  data.yaml: {data_yaml}")
    print("=" * 60)


if __name__ == "__main__":
    main()
