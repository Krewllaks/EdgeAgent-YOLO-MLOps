"""
V4 Dataset Preparation — TUM veri kaynaklari birlesik.

Kaynak 1: V1 base dataset (717 train, 89 val, 91 test) — 3 sinif etiketli
Kaynak 2: erdogan1/model train (1194 goruntu) — screw (class 0) etiketli
Kaynak 3: erdogan1/NOK fotograflar (905 goruntu) — etiketsiz, background olarak
Kaynak 4: Tum crop kaynaklari (coklanmis, coklanmisacili, coklanmisyeni, coklanmis1000)
           -> copy-paste augmentation icin missing_screw + missing_component crop'lari

NOT: erdogan2 = erdogan1 ile ayni (duplicate), atlanir.
NOT: roboflowetiketlenen = V1'in kaynagi, zaten dahil.
NOT: Screw crop'lari ATLANIR (erdogan1 + V1'den yeterli screw var).

Kullanim:
    python scripts/prepare_v4_dataset.py
    python scripts/prepare_v4_dataset.py --num-augmented 2000
    python scripts/prepare_v4_dataset.py --dry-run
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

# Reuse copy-paste logic from V3
from scripts.prepare_v3_copypaste import (
    load_crops,
    find_screw_positions,
    generate_augmented_image,
    md5_file,
    IMAGE_EXTS,
    CLASS_NAMES,
)

# All crop sources (for copy-paste augmentation)
CROP_SOURCES = [
    ROOT / "coklanmis1000",
    ROOT / "coklanmisyeni",
    ROOT / "coklanmis",
    ROOT / "coklanmisacili",
]

# Extra labeled data (screw-only, class 0)
ERDOGAN_YOLO = ROOT / "erdogan1" / "model"

# NOK photos (unlabeled, used as extra backgrounds for copy-paste)
NOK_SOURCES = [
    ROOT / "erdogan1" / "NOK fotoğraflar" / "defected_camera1",
    ROOT / "erdogan1" / "NOK fotoğraflar" / "defected_camera2",
]


def load_all_crops() -> tuple[list, list]:
    """Load missing_screw and missing_component crops from ALL sources."""
    all_ms_crops = []
    all_mc_crops = []

    for source_dir in CROP_SOURCES:
        if not source_dir.exists():
            print(f"  [SKIP] {source_dir.name} bulunamadi")
            continue

        ms_dir = source_dir / "missing_screw"
        mc_dir = source_dir / "missing_component"

        ms = load_crops(ms_dir, class_id=1) if ms_dir.exists() else []
        mc = load_crops(mc_dir, class_id=2) if mc_dir.exists() else []

        print(f"  {source_dir.name}: missing_screw={len(ms)}, missing_component={len(mc)}")
        all_ms_crops.extend(ms)
        all_mc_crops.extend(mc)

    return all_ms_crops, all_mc_crops


def collect_erdogan_train(valtest_hashes: set) -> tuple[list, list]:
    """Collect erdogan1/model train images + labels (screw-only, class 0).

    Returns:
        (image_paths, label_paths) — matched pairs, leakage-checked
    """
    img_dir = ERDOGAN_YOLO / "train" / "images"
    lbl_dir = ERDOGAN_YOLO / "train" / "labels"

    if not img_dir.exists():
        print("  [SKIP] erdogan1/model/train bulunamadi")
        return [], []

    img_paths = []
    lbl_paths = []
    skipped_leak = 0

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        # Leakage check
        h = md5_file(img_path)
        if h in valtest_hashes:
            skipped_leak += 1
            continue

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        img_paths.append(img_path)
        lbl_paths.append(lbl_path if lbl_path.exists() else None)

    print(f"  erdogan1/model train: {len(img_paths)} goruntu (leakage skip: {skipped_leak})")
    return img_paths, lbl_paths


def collect_nok_backgrounds(valtest_hashes: set) -> list:
    """Collect NOK photos as extra backgrounds for copy-paste augmentation."""
    backgrounds = []

    for nok_dir in NOK_SOURCES:
        if not nok_dir.exists():
            print(f"  [SKIP] {nok_dir.name} bulunamadi")
            continue

        count = 0
        for img_path in sorted(nok_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            # Leakage check
            h = md5_file(img_path)
            if h in valtest_hashes:
                continue
            backgrounds.append(img_path)
            count += 1

        print(f"  NOK {nok_dir.name}: {count} background goruntu")

    return backgrounds


def main():
    parser = argparse.ArgumentParser(description="V4 Dataset — Tum kaynaklar birlesik")
    parser.add_argument(
        "--base-dataset", type=Path,
        default=ROOT / "data" / "processed" / "phase1_multiclass_v1",
        help="Base V1 dataset",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "processed" / "phase1_v4",
        help="Output V4 dataset directory",
    )
    parser.add_argument("--num-augmented", type=int, default=2000,
                        help="Number of augmented images to generate (default: 2000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("=" * 60)
    print("  V4 Dataset Preparation — TUM Kaynaklar Birlesik")
    print("=" * 60)

    # ── Step 1: Load ALL crops ──────────────────────────────────
    print("\n[1/6] Tum kaynaklardan crop'lar yukleniyor...")
    ms_crops, mc_crops = load_all_crops()
    print(f"\n  TOPLAM missing_screw crops : {len(ms_crops)}")
    print(f"  TOPLAM missing_component crops: {len(mc_crops)}")

    if not ms_crops and not mc_crops:
        print("[ERROR] Hic crop bulunamadi!")
        sys.exit(1)

    # Deduplicate crops by hash
    print("\n  Crop dedup (MD5)...")
    seen_hashes = set()
    unique_ms = []
    for crop in ms_crops:
        h = hashlib.md5(crop["image"].tobytes()[:2048]).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_ms.append(crop)

    unique_mc = []
    for crop in mc_crops:
        h = hashlib.md5(crop["image"].tobytes()[:2048]).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_mc.append(crop)

    print(f"  Dedup sonrasi: missing_screw={len(unique_ms)} (was {len(ms_crops)})")
    print(f"  Dedup sonrasi: missing_component={len(unique_mc)} (was {len(mc_crops)})")
    ms_crops = unique_ms
    mc_crops = unique_mc

    # ── Step 2: Load base datasets ─────────────────────────────
    print("\n[2/6] Veri kaynaklari yukleniyor...")
    base_train_img = args.base_dataset / "train" / "images"
    base_train_lbl = args.base_dataset / "train" / "labels"
    base_val_img = args.base_dataset / "val" / "images"
    base_val_lbl = args.base_dataset / "val" / "labels"
    base_test_img = args.base_dataset / "test" / "images"
    base_test_lbl = args.base_dataset / "test" / "labels"

    train_images = sorted([p for p in base_train_img.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    print(f"  V1 base train: {len(train_images)} goruntu")

    # Val/test hashes for leakage prevention
    valtest_hashes = set()
    for split_dir in [base_val_img, base_test_img]:
        if split_dir.exists():
            for p in split_dir.iterdir():
                if p.suffix.lower() in IMAGE_EXTS:
                    valtest_hashes.add(md5_file(p))
    print(f"  Val/Test hashes: {len(valtest_hashes)}")

    # ── Step 3: Collect erdogan1 train (screw-only labels) ─────
    print("\n[3/6] erdogan1/model train yukleniyor...")
    erdogan_imgs, erdogan_lbls = collect_erdogan_train(valtest_hashes)

    # ── Step 4: Collect NOK backgrounds ────────────────────────
    print("\n[4/6] NOK fotograflar (background) yukleniyor...")
    nok_backgrounds = collect_nok_backgrounds(valtest_hashes)

    # Combine all backgrounds for augmentation
    # V1 train images (have screw labels -> good for copy-paste)
    # NOK backgrounds (defective photos -> paste defects onto them)
    all_bg_for_augment = list(train_images)  # V1 train images with screw labels

    if args.dry_run:
        total_direct = len(train_images) + len(erdogan_imgs)
        print(f"\n[DRY RUN] Ozet:")
        print(f"  V1 base train: {len(train_images)}")
        print(f"  erdogan1 train (screw labels): {len(erdogan_imgs)}")
        print(f"  NOK backgrounds: {len(nok_backgrounds)}")
        print(f"  Toplam dogrudan train: {total_direct}")
        print(f"  Crop havuzu: {len(ms_crops) + len(mc_crops)} (ms={len(ms_crops)}, mc={len(mc_crops)})")
        print(f"  Augmented uretilecek: {args.num_augmented}")
        print(f"  Tahmini final train: {total_direct + args.num_augmented}")
        print(f"  Output: {args.output}")
        return

    # ── Step 5: Generate augmented images ──────────────────────
    print(f"\n[5/6] {args.num_augmented} augmented goruntu uretiliyor...")
    augmented_pairs = []

    for i in range(args.num_augmented):
        # 70% V1 backgrounds (have screw labels), 30% NOK backgrounds
        use_nok = nok_backgrounds and rng.random() < 0.30

        if use_nok:
            # NOK background - no screw labels, paste anywhere
            bg_path = rng.choice(nok_backgrounds)
            bg_img = cv2.imread(str(bg_path))
            if bg_img is None:
                continue

            h, w = bg_img.shape[:2]
            # Create synthetic screw positions (4 positions on typical locations)
            # Based on domain: 2 left side, 2 right side
            synthetic_labels = [
                f"0 0.20 0.35 0.06 0.08",
                f"0 0.20 0.65 0.06 0.08",
                f"0 0.80 0.35 0.06 0.08",
                f"0 0.80 0.65 0.06 0.08",
            ]
            bg_labels = synthetic_labels
        else:
            # V1 background with real screw labels
            base_img_path = rng.choice(train_images)
            base_lbl_path = base_train_lbl / f"{base_img_path.stem}.txt"

            bg_img = cv2.imread(str(base_img_path))
            if bg_img is None:
                continue

            bg_labels = find_screw_positions(base_lbl_path)
            if not bg_labels:
                continue

        aug_img, aug_labels = generate_augmented_image(
            bg_img, bg_labels, ms_crops, mc_crops, rng,
        )

        augmented_pairs.append({
            "image": aug_img,
            "labels": aug_labels,
            "source": bg_path.name if use_nok else base_img_path.name,
            "index": i,
        })

        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{args.num_augmented}")

    print(f"  Uretilen: {len(augmented_pairs)}")

    # ── Step 6: Build V4 dataset ───────────────────────────────
    print(f"\n[6/6] V4 dataset olusturuluyor...")
    output = args.output
    if output.exists():
        shutil.rmtree(output)

    for split in ["train", "val", "test"]:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)

    # ── 6a: Copy original V1 train/val/test ────────────────────
    copied = {"train": 0, "val": 0, "test": 0}
    seen_train_hashes = set()

    for split, src_img_dir, src_lbl_dir in [
        ("train", base_train_img, base_train_lbl),
        ("val", base_val_img, base_val_lbl),
        ("test", base_test_img, base_test_lbl),
    ]:
        if not src_img_dir.exists():
            continue
        for img_path in src_img_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            shutil.copy2(img_path, output / split / "images" / img_path.name)
            src_lbl = src_lbl_dir / f"{img_path.stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, output / split / "labels" / f"{img_path.stem}.txt")
            if split == "train":
                seen_train_hashes.add(md5_file(img_path))
            copied[split] += 1

    print(f"  V1 kopyalanan: train={copied['train']}, val={copied['val']}, test={copied['test']}")

    # ── 6b: Add erdogan1/model train images (screw labels) ────
    erdogan_added = 0
    for img_path, lbl_path in zip(erdogan_imgs, erdogan_lbls):
        # MD5 dedup — skip if already in train from V1
        h = md5_file(img_path)
        if h in seen_train_hashes:
            continue
        seen_train_hashes.add(h)

        dst_img = output / "train" / "images" / img_path.name
        shutil.copy2(img_path, dst_img)

        if lbl_path and lbl_path.exists():
            # erdogan labels are class 0 (vida-ok = screw), same mapping
            shutil.copy2(lbl_path, output / "train" / "labels" / f"{img_path.stem}.txt")
        erdogan_added += 1

    print(f"  erdogan1/model eklenen: {erdogan_added} (dedup skip: {len(erdogan_imgs) - erdogan_added})")

    # ── 6c: Add augmented images ───────────────────────────────
    aug_added = 0
    aug_class_counts = {0: 0, 1: 0, 2: 0}

    for pair in augmented_pairs:
        img_hash = hashlib.md5(pair["image"].tobytes()[:2048]).hexdigest()[:10]
        filename = f"v4_cpaste_{pair['index']:04d}_{img_hash}"

        dst_img = output / "train" / "images" / f"{filename}.jpg"
        dst_lbl = output / "train" / "labels" / f"{filename}.txt"

        cv2.imwrite(str(dst_img), pair["image"])
        dst_lbl.write_text("\n".join(pair["labels"]), encoding="utf-8")

        for lbl_line in pair["labels"]:
            cls_id = int(lbl_line.split()[0])
            aug_class_counts[cls_id] = aug_class_counts.get(cls_id, 0) + 1

        aug_added += 1

    print(f"  Augmented eklenen: {aug_added}")
    print(f"\n  Augmented sinif dagilimi:")
    for cls_id in sorted(aug_class_counts.keys()):
        print(f"    {CLASS_NAMES.get(cls_id, f'cls_{cls_id}')}: {aug_class_counts[cls_id]}")

    # ── Final distribution ─────────────────────────────────────
    final_counts = {0: 0, 1: 0, 2: 0}
    lbl_dir = output / "train" / "labels"
    for lbl_path in lbl_dir.glob("*.txt"):
        for line in lbl_path.read_text(encoding="utf-8").strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                final_counts[cls_id] = final_counts.get(cls_id, 0) + 1

    total_boxes = sum(final_counts.values())
    print(f"\n  Final train sinif dagilimi (tum bboxlar):")
    for cls_id in sorted(final_counts.keys()):
        name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
        count = final_counts[cls_id]
        pct = count / total_boxes * 100 if total_boxes > 0 else 0
        print(f"    {name}: {count} ({pct:.1f}%)")

    # Write data.yaml
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
    total_train = copied["train"] + erdogan_added + aug_added
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "version": "v4",
        "base_dataset": str(args.base_dataset),
        "crop_sources": [str(s) for s in CROP_SOURCES],
        "erdogan_source": str(ERDOGAN_YOLO),
        "nok_backgrounds": [str(s) for s in NOK_SOURCES],
        "num_v1_train": copied["train"],
        "num_erdogan_train": erdogan_added,
        "num_augmented": aug_added,
        "unique_ms_crops": len(ms_crops),
        "unique_mc_crops": len(mc_crops),
        "nok_bg_count": len(nok_backgrounds),
        "splits": {"train": total_train, "val": copied["val"], "test": copied["test"]},
        "final_class_distribution": {CLASS_NAMES[k]: v for k, v in final_counts.items()},
        "augmented_class_distribution": {CLASS_NAMES[k]: v for k, v in aug_class_counts.items()},
        "seed": args.seed,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"\n  TAMAMLANDI!")
    print(f"  Output    : {output}")
    print(f"  data.yaml : {data_yaml}")
    print(f"  Train     : {total_train}")
    print(f"    - V1 orijinal  : {copied['train']}")
    print(f"    - erdogan1     : {erdogan_added}")
    print(f"    - augmented    : {aug_added}")
    print(f"  Val       : {copied['val']}")
    print(f"  Test      : {copied['test']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
