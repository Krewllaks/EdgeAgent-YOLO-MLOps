"""
V3 Dataset Preparation — Copy-Paste Augmentation.

Muhammet'in gonderdigi 128x128 crop'lari (missing_screw, missing_component)
mevcut V1 train goruntulerinin uzerine yapistirarak yeni labeled data uretir.

Strateji:
1. V1 train setini baz al (717 goruntu, temiz label'lar)
2. missing_screw ve missing_component crop'larini vida pozisyonlarina yapistir
3. Label otomatik hesaplanir (yapistirma koordinatlarindan)
4. 80/10/10 split ile V3 dataset olustur

Kullanim:
    python scripts/prepare_v3_copypaste.py
    python scripts/prepare_v3_copypaste.py --num-augmented 400 --dry-run
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASS_NAMES = {0: "screw", 1: "missing_screw", 2: "missing_component"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_crops(crop_dir: Path, class_id: int) -> list[dict]:
    """Load crop images from a directory."""
    crops = []
    if not crop_dir.exists():
        return crops

    for img_path in sorted(crop_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        crops.append({
            "image": img,
            "class_id": class_id,
            "source": img_path.name,
            "h": img.shape[0],
            "w": img.shape[1],
        })
    return crops


def find_screw_positions(label_path: Path) -> list[dict]:
    """Extract screw bbox positions from a label file."""
    positions = []
    if not label_path.exists():
        return positions

    for line in label_path.read_text(encoding="utf-8").strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        cx, cy, w, h = [float(x) for x in parts[1:5]]
        positions.append({
            "class_id": class_id,
            "cx": cx, "cy": cy, "w": w, "h": h,
        })
    return positions


def paste_crop_on_image(
    background: np.ndarray,
    crop: np.ndarray,
    target_cx: float,
    target_cy: float,
    scale_range: tuple = (0.8, 1.5),
    rng: random.Random = None,
) -> tuple[np.ndarray, list[float]]:
    """Paste a crop onto a background image at a target position.

    Returns: (modified_image, [cx, cy, w, h] in normalized coords)
    """
    if rng is None:
        rng = random.Random()

    bg_h, bg_w = background.shape[:2]
    crop_h, crop_w = crop.shape[:2]

    # Scale crop to match typical screw bbox size (~3.5-5% of image)
    # Target bbox is typically ~30-40 pixels on a 800x600 image
    target_pixel_w = int(bg_w * rng.uniform(0.035, 0.055))
    target_pixel_h = int(bg_h * rng.uniform(0.040, 0.065))

    # Apply random scale variation
    scale = rng.uniform(*scale_range)
    final_w = max(10, int(target_pixel_w * scale))
    final_h = max(10, int(target_pixel_h * scale))

    # Resize crop
    crop_resized = cv2.resize(crop, (final_w, final_h), interpolation=cv2.INTER_AREA)

    # Calculate paste position (centered on target)
    paste_x = int(target_cx * bg_w - final_w / 2)
    paste_y = int(target_cy * bg_h - final_h / 2)

    # Add small random offset for variation
    paste_x += rng.randint(-5, 5)
    paste_y += rng.randint(-5, 5)

    # Clamp to image bounds
    paste_x = max(0, min(bg_w - final_w, paste_x))
    paste_y = max(0, min(bg_h - final_h, paste_y))

    # Paste with slight alpha blending at edges
    result = background.copy()

    # Create a smooth blending mask
    mask = np.ones((final_h, final_w), dtype=np.float32)
    border = max(2, min(final_w, final_h) // 6)
    for i in range(border):
        alpha = (i + 1) / border
        mask[i, :] *= alpha
        mask[-(i + 1), :] *= alpha
        mask[:, i] *= alpha
        mask[:, -(i + 1)] *= alpha
    mask = mask[:, :, np.newaxis]

    # Blend
    roi = result[paste_y:paste_y + final_h, paste_x:paste_x + final_w]
    blended = (crop_resized.astype(np.float32) * mask +
               roi.astype(np.float32) * (1 - mask))
    result[paste_y:paste_y + final_h, paste_x:paste_x + final_w] = blended.astype(np.uint8)

    # Calculate YOLO bbox (normalized)
    bbox_cx = (paste_x + final_w / 2) / bg_w
    bbox_cy = (paste_y + final_h / 2) / bg_h
    bbox_w = final_w / bg_w
    bbox_h = final_h / bg_h

    return result, [bbox_cx, bbox_cy, bbox_w, bbox_h]


def generate_augmented_image(
    bg_img: np.ndarray,
    bg_labels: list[dict],
    missing_screw_crops: list[dict],
    missing_component_crops: list[dict],
    rng: random.Random,
    strategy: str = "random",
) -> tuple[np.ndarray, list[str]]:
    """Generate one augmented image by replacing some screws with defects.

    Strategy:
    - Pick 1-2 random screw positions
    - Replace with missing_screw or missing_component crop
    - Keep remaining screws as-is
    """
    result = bg_img.copy()
    new_labels = []

    # Decide how many screws to replace (1 or 2)
    screw_positions = [p for p in bg_labels if p["class_id"] == 0]
    other_positions = [p for p in bg_labels if p["class_id"] != 0]

    if not screw_positions:
        # No screws to replace, keep original labels
        for p in bg_labels:
            new_labels.append(f"{p['class_id']} {p['cx']:.6f} {p['cy']:.6f} {p['w']:.6f} {p['h']:.6f}")
        return result, new_labels

    # Randomly select 1-2 positions to replace
    num_replace = rng.randint(1, min(2, len(screw_positions)))
    replace_indices = rng.sample(range(len(screw_positions)), num_replace)

    for idx, pos in enumerate(screw_positions):
        if idx in replace_indices:
            # Replace this screw with a defect
            if rng.random() < 0.55:
                # missing_screw (slightly more common)
                crop_info = rng.choice(missing_screw_crops)
                new_class_id = 1
            else:
                # missing_component
                crop_info = rng.choice(missing_component_crops)
                new_class_id = 2

            # Paste crop at this screw position
            result, bbox = paste_crop_on_image(
                result, crop_info["image"],
                target_cx=pos["cx"], target_cy=pos["cy"],
                scale_range=(0.8, 1.3),
                rng=rng,
            )
            new_labels.append(f"{new_class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
        else:
            # Keep original screw
            new_labels.append(f"{pos['class_id']} {pos['cx']:.6f} {pos['cy']:.6f} {pos['w']:.6f} {pos['h']:.6f}")

    # Keep other original detections (missing_screw, missing_component if any)
    for p in other_positions:
        new_labels.append(f"{p['class_id']} {p['cx']:.6f} {p['cy']:.6f} {p['w']:.6f} {p['h']:.6f}")

    return result, new_labels


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes()[:4096])
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="V3 Copy-Paste Augmentation Dataset")
    parser.add_argument(
        "--crop-dir", type=Path,
        default=ROOT / "coklanmisyeni",
        help="Directory with crop folders (missing_screw/, missing_component/)",
    )
    parser.add_argument(
        "--base-dataset", type=Path,
        default=ROOT / "data" / "processed" / "phase1_multiclass_v1",
        help="Base dataset to augment from",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "processed" / "phase1_v3",
        help="Output V3 dataset directory",
    )
    parser.add_argument("--num-augmented", type=int, default=500,
                        help="Number of augmented images to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", type=str, default="80/10/10",
                        help="Train/Val/Test split ratio")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)

    # Parse split ratio
    splits = [int(x) for x in args.split_ratio.split("/")]
    assert len(splits) == 3 and sum(splits) == 100

    print("=" * 60)
    print("  V3 Dataset Preparation — Copy-Paste Augmentation")
    print("=" * 60)

    # Load crops (skip screw — already have enough)
    print("\n[1/5] Loading crops...")
    ms_crops = load_crops(args.crop_dir / "missing_screw", class_id=1)
    mc_crops = load_crops(args.crop_dir / "missing_component", class_id=2)
    print(f"  missing_screw crops : {len(ms_crops)}")
    print(f"  missing_component crops: {len(mc_crops)}")

    if not ms_crops and not mc_crops:
        print("[ERROR] No crops found!")
        sys.exit(1)

    # Load base dataset
    print("\n[2/5] Loading base dataset...")
    base_train_img = args.base_dataset / "train" / "images"
    base_train_lbl = args.base_dataset / "train" / "labels"
    base_val_img = args.base_dataset / "val" / "images"
    base_val_lbl = args.base_dataset / "val" / "labels"
    base_test_img = args.base_dataset / "test" / "images"
    base_test_lbl = args.base_dataset / "test" / "labels"

    train_images = sorted([p for p in base_train_img.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    print(f"  Base train images: {len(train_images)}")

    # Collect val/test hashes for leakage prevention
    valtest_hashes = set()
    for split_dir in [base_val_img, base_test_img]:
        if split_dir.exists():
            for p in split_dir.iterdir():
                if p.suffix.lower() in IMAGE_EXTS:
                    valtest_hashes.add(md5_file(p))
    print(f"  Val/Test hashes for leakage check: {len(valtest_hashes)}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would generate {args.num_augmented} augmented images")
        print(f"  Output: {args.output}")
        return

    # Generate augmented images
    print(f"\n[3/5] Generating {args.num_augmented} augmented images...")
    augmented_pairs = []  # (image, labels_list)

    for i in range(args.num_augmented):
        # Pick random base image
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
            "source": base_img_path.name,
            "index": i,
        })

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{args.num_augmented}")

    print(f"  Generated: {len(augmented_pairs)}")

    # Split: augmented images go to train, originals stay in their splits
    print(f"\n[4/5] Building V3 dataset...")

    output = args.output
    for split in ["train", "val", "test"]:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy original V1 train/val/test as-is
    copied = {"train": 0, "val": 0, "test": 0}
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
            dst_img = output / split / "images" / img_path.name
            dst_lbl = output / split / "labels" / f"{img_path.stem}.txt"
            src_lbl = src_lbl_dir / f"{img_path.stem}.txt"

            shutil.copy2(img_path, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            copied[split] += 1

    # Add augmented images to train
    aug_added = 0
    class_counts = {0: 0, 1: 0, 2: 0}

    for pair in augmented_pairs:
        img_hash = hashlib.md5(pair["image"].tobytes()[:2048]).hexdigest()[:10]
        filename = f"cpaste_{pair['index']:04d}_{img_hash}"

        dst_img = output / "train" / "images" / f"{filename}.jpg"
        dst_lbl = output / "train" / "labels" / f"{filename}.txt"

        cv2.imwrite(str(dst_img), pair["image"])
        dst_lbl.write_text("\n".join(pair["labels"]), encoding="utf-8")

        # Count classes
        for lbl_line in pair["labels"]:
            cls_id = int(lbl_line.split()[0])
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        aug_added += 1

    print(f"  Original copied: train={copied['train']}, val={copied['val']}, test={copied['test']}")
    print(f"  Augmented added to train: {aug_added}")
    print(f"  Augmented class distribution:")
    for cls_id, count in sorted(class_counts.items()):
        print(f"    {CLASS_NAMES.get(cls_id, f'cls_{cls_id}')}: {count}")

    # Count final distribution
    print(f"\n  Final train label distribution:")
    final_counts = {0: 0, 1: 0, 2: 0}
    lbl_dir = output / "train" / "labels"
    for lbl_path in lbl_dir.glob("*.txt"):
        for line in lbl_path.read_text(encoding="utf-8").strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                final_counts[cls_id] = final_counts.get(cls_id, 0) + 1

    total_boxes = sum(final_counts.values())
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

    # Write summary
    total_train = copied["train"] + aug_added
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_dataset": str(args.base_dataset),
        "crop_source": str(args.crop_dir),
        "num_augmented": aug_added,
        "splits": {
            "train": total_train,
            "val": copied["val"],
            "test": copied["test"],
        },
        "final_class_distribution": {CLASS_NAMES[k]: v for k, v in final_counts.items()},
        "augmented_class_distribution": {CLASS_NAMES[k]: v for k, v in class_counts.items()},
        "seed": args.seed,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"\n[5/5] Done!")
    print(f"  Output: {output}")
    print(f"  data.yaml: {data_yaml}")
    print(f"  Train: {total_train} images (original: {copied['train']} + augmented: {aug_added})")
    print(f"  Val: {copied['val']} images")
    print(f"  Test: {copied['test']} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
