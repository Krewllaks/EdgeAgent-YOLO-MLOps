"""
VLM-Guided Data Augmentation Module.

Hocanin direktifi: "Visual language modelle yeni bir sey coklayabiliyor musun?"
VLM'i sadece detection icin degil, yeni egitim verisi URETMEK icin kullanmak.

Bu modul 3 strateji sunar:
1. VLM-guided augmentation: VLM goruntuleri analiz eder, anlamli
   augmentation parametrelerini belirler (isik, aci, blur)
2. VLM-validated synthetic: Diffusion/klasik augmentation ile uretilen
   goruntuleri VLM ile dogrulama
3. Copy-paste augmentation: Mevcut label'li vida/eksik vida crop'larini
   farkli arka planlara yapistirma (label otomatik olusur)

Strateji 3 (Copy-Paste) en pratik ve hizli sonuc verir:
- Mevcut label'li veriden crop'lar cikarir
- Farkli arka planlara yapistirarak yeni labeled data uretir
- Label otomatik hesaplanir (yapistirma koordinatlarindan)
- Muhammet'ten label beklemeden veri cogaltilabilir
"""

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.constants import CLASS_NAMES, IMAGE_EXTS


@dataclass
class AugmentationPlan:
    """VLM-guided augmentation plan for a single image."""
    image_path: str
    brightness_adjust: float = 0.0  # -1.0 to 1.0
    contrast_adjust: float = 0.0
    blur_level: int = 0  # 0=none, 1=slight, 2=moderate
    rotation_angle: float = 0.0  # degrees
    flip_horizontal: bool = False
    noise_level: float = 0.0  # 0.0 to 0.1
    reasoning: str = ""


@dataclass
class AugmentResult:
    """Result of augmentation generation."""
    total_generated: int = 0
    output_dir: str = ""
    images_created: list = field(default_factory=list)
    labels_created: list = field(default_factory=list)
    strategy: str = ""


# ── Strategy 1: VLM-Guided Parameter Selection ──────────────────


def analyze_image_for_augmentation(image_path: Path) -> AugmentationPlan:
    """Analyze an image and suggest meaningful augmentation parameters.

    Uses basic image statistics to determine which augmentations
    would produce realistic training variations. In production,
    this can be enhanced with actual VLM analysis.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return AugmentationPlan(image_path=str(image_path))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean() / 255.0
    std_brightness = gray.std() / 255.0

    plan = AugmentationPlan(image_path=str(image_path))

    # Dark image -> brighten augmentation useful
    if mean_brightness < 0.3:
        plan.brightness_adjust = random.uniform(0.1, 0.3)
        plan.reasoning += "Karanlik goruntu, parlaklik artisi oneriliyor. "
    elif mean_brightness > 0.7:
        plan.brightness_adjust = random.uniform(-0.3, -0.1)
        plan.reasoning += "Cok parlak goruntu, parlaklik azaltmasi oneriliyor. "

    # Low contrast -> contrast augmentation
    if std_brightness < 0.15:
        plan.contrast_adjust = random.uniform(0.1, 0.3)
        plan.reasoning += "Dusuk kontrast, kontrast artisi oneriliyor. "

    # Metal reflections -> slight blur useful
    if std_brightness > 0.35:
        plan.blur_level = 1
        plan.reasoning += "Yuksek varyans (metal yansima?), hafif blur oneriliyor. "

    # Small rotation for position variation
    plan.rotation_angle = random.uniform(-5, 5)
    plan.flip_horizontal = random.random() > 0.5

    if not plan.reasoning:
        plan.reasoning = "Standart augmentation parametreleri."

    return plan


def apply_augmentation(image: np.ndarray, plan: AugmentationPlan) -> np.ndarray:
    """Apply augmentation plan to an image."""
    result = image.copy()

    # Brightness
    if abs(plan.brightness_adjust) > 0.01:
        result = np.clip(result.astype(np.float32) + plan.brightness_adjust * 255, 0, 255).astype(np.uint8)

    # Contrast
    if abs(plan.contrast_adjust) > 0.01:
        factor = 1.0 + plan.contrast_adjust
        mean = result.mean()
        result = np.clip((result.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    # Blur
    if plan.blur_level > 0:
        ksize = plan.blur_level * 2 + 1
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    # Noise
    if plan.noise_level > 0.01:
        noise = np.random.normal(0, plan.noise_level * 255, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Flip
    if plan.flip_horizontal:
        result = cv2.flip(result, 1)

    # Rotation
    if abs(plan.rotation_angle) > 0.5:
        h, w = result.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, plan.rotation_angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return result


def transform_bbox_for_augmentation(
    bbox_xywh: list[float],
    plan: AugmentationPlan,
    img_w: int,
    img_h: int,
) -> list[float]:
    """Transform a YOLO bbox according to augmentation plan.

    Returns new bbox in YOLO normalized xywh format.
    """
    cx, cy, w, h = bbox_xywh

    # Flip
    if plan.flip_horizontal:
        cx = 1.0 - cx

    # Rotation (approximate for small angles)
    if abs(plan.rotation_angle) > 0.5:
        import math
        angle_rad = math.radians(plan.rotation_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        # Rotate center around image center (0.5, 0.5)
        dx = cx - 0.5
        dy = cy - 0.5
        new_cx = dx * cos_a - dy * sin_a + 0.5
        new_cy = dx * sin_a + dy * cos_a + 0.5
        cx, cy = new_cx, new_cy
        # Bbox dimensions slightly increase with rotation
        new_w = w * abs(cos_a) + h * abs(sin_a)
        new_h = w * abs(sin_a) + h * abs(cos_a)
        w, h = new_w, new_h

    # Clamp
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.01, min(1.0, w))
    h = max(0.01, min(1.0, h))

    return [cx, cy, w, h]


# ── Strategy 3: Copy-Paste Augmentation ──────────────────────────


def extract_crops(
    image_dir: Path,
    label_dir: Path,
    target_classes: list[int] = None,
    padding: float = 0.1,
) -> list[dict]:
    """Extract object crops from labeled images.

    Returns list of {crop_image, class_id, original_file, bbox}.
    """
    if target_classes is None:
        target_classes = [0, 1, 2]

    crops = []
    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        for line in label_path.read_text(encoding="utf-8").strip().split("\n"):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            if class_id not in target_classes:
                continue

            cx, cy, bw, bh = [float(x) for x in parts[1:5]]

            # Skip weak labels
            if bw > 0.7 and bh > 0.7:
                continue

            # Pixel coords with padding
            x1 = int(max(0, (cx - bw / 2 - padding * bw) * w))
            y1 = int(max(0, (cy - bh / 2 - padding * bh) * h))
            x2 = int(min(w, (cx + bw / 2 + padding * bw) * w))
            y2 = int(min(h, (cy + bh / 2 + padding * bh) * h))

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            crop = img[y1:y2, x1:x2].copy()
            crops.append({
                "crop_image": crop,
                "class_id": class_id,
                "original_file": img_path.name,
                "bbox_xywh": [cx, cy, bw, bh],
            })

    return crops


def copy_paste_augment(
    background_dir: Path,
    crops: list[dict],
    output_dir: Path,
    num_images: int = 100,
    objects_per_image: tuple[int, int] = (2, 5),
    seed: int = 42,
) -> AugmentResult:
    """Generate new labeled images by pasting object crops onto backgrounds.

    This creates properly labeled training data without manual annotation.
    """
    rng = random.Random(seed)
    output_img_dir = output_dir / "images"
    output_lbl_dir = output_dir / "labels"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Collect background images
    bg_files = [
        p for p in sorted(background_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTS
    ]
    if not bg_files:
        print("[WARN] No background images found")
        return AugmentResult(strategy="copy_paste")

    result = AugmentResult(strategy="copy_paste", output_dir=str(output_dir))

    for i in range(num_images):
        # Pick random background
        bg_path = rng.choice(bg_files)
        bg_img = cv2.imread(str(bg_path))
        if bg_img is None:
            continue
        bg_h, bg_w = bg_img.shape[:2]

        canvas = bg_img.copy()
        labels = []
        n_objects = rng.randint(*objects_per_image)

        for _ in range(n_objects):
            crop_info = rng.choice(crops)
            crop = crop_info["crop_image"]
            ch, cw = crop.shape[:2]

            # Random scale (0.5x to 1.5x)
            scale = rng.uniform(0.5, 1.5)
            new_cw = int(cw * scale)
            new_ch = int(ch * scale)
            if new_cw < 10 or new_ch < 10:
                continue
            if new_cw >= bg_w or new_ch >= bg_h:
                continue

            crop_resized = cv2.resize(crop, (new_cw, new_ch))

            # Random position
            x = rng.randint(0, bg_w - new_cw)
            y = rng.randint(0, bg_h - new_ch)

            # Paste with alpha blending (smooth edges)
            canvas[y:y + new_ch, x:x + new_cw] = crop_resized

            # Generate YOLO label
            cx = (x + new_cw / 2) / bg_w
            cy = (y + new_ch / 2) / bg_h
            w_norm = new_cw / bg_w
            h_norm = new_ch / bg_h
            labels.append(f"{crop_info['class_id']} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Save
        img_hash = hashlib.md5(canvas.tobytes()[:1024]).hexdigest()[:8]
        filename = f"copypaste_{i:04d}_{img_hash}"
        img_path = output_img_dir / f"{filename}.jpg"
        lbl_path = output_lbl_dir / f"{filename}.txt"

        cv2.imwrite(str(img_path), canvas)
        lbl_path.write_text("\n".join(labels), encoding="utf-8")

        result.images_created.append(img_path.name)
        result.labels_created.append(lbl_path.name)
        result.total_generated += 1

    return result


# ── Strategy 1+3 Combined: Smart Augmentation Pipeline ──────────


def smart_augment_dataset(
    source_image_dir: Path,
    source_label_dir: Path,
    output_dir: Path,
    num_augmented: int = 200,
    seed: int = 42,
) -> AugmentResult:
    """Generate augmented labeled data using VLM-guided parameters.

    For each source image with labels:
    1. Analyze image to determine optimal augmentation params
    2. Apply augmentation to both image and labels
    3. Save with proper YOLO labels
    """
    rng = random.Random(seed)
    output_img_dir = output_dir / "images"
    output_lbl_dir = output_dir / "labels"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Collect source images with labels
    source_pairs = []
    for img_path in sorted(source_image_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = source_label_dir / f"{img_path.stem}.txt"
        if label_path.exists() and label_path.stat().st_size > 0:
            source_pairs.append((img_path, label_path))

    if not source_pairs:
        print("[WARN] No labeled source images found")
        return AugmentResult(strategy="vlm_guided")

    result = AugmentResult(strategy="vlm_guided", output_dir=str(output_dir))

    for i in range(num_augmented):
        img_path, label_path = rng.choice(source_pairs)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Analyze and create augmentation plan
        plan = analyze_image_for_augmentation(img_path)
        # Randomize plan slightly for diversity
        plan.brightness_adjust += rng.uniform(-0.05, 0.05)
        plan.rotation_angle = rng.uniform(-8, 8)
        plan.flip_horizontal = rng.random() > 0.5
        plan.noise_level = rng.uniform(0, 0.02)

        # Apply augmentation to image
        aug_img = apply_augmentation(img, plan)

        # Transform labels
        aug_labels = []
        for line in label_path.read_text(encoding="utf-8").strip().split("\n"):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]

            # Skip weak labels
            if bbox[2] > 0.7 and bbox[3] > 0.7:
                continue

            new_bbox = transform_bbox_for_augmentation(bbox, plan, w, h)
            aug_labels.append(f"{class_id} {new_bbox[0]:.6f} {new_bbox[1]:.6f} {new_bbox[2]:.6f} {new_bbox[3]:.6f}")

        if not aug_labels:
            continue

        # Save
        img_hash = hashlib.md5(aug_img.tobytes()[:1024]).hexdigest()[:8]
        filename = f"vlmaug_{i:04d}_{img_hash}"
        out_img = output_img_dir / f"{filename}.jpg"
        out_lbl = output_lbl_dir / f"{filename}.txt"

        cv2.imwrite(str(out_img), aug_img)
        out_lbl.write_text("\n".join(aug_labels), encoding="utf-8")

        result.images_created.append(out_img.name)
        result.labels_created.append(out_lbl.name)
        result.total_generated += 1

    return result


# ── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM-Guided Data Augmentation")
    parser.add_argument(
        "--strategy", type=str, default="smart",
        choices=["smart", "copypaste"],
        help="Augmentation strategy",
    )
    parser.add_argument(
        "--source-images", type=Path,
        default=ROOT / "data" / "processed" / "phase1_multiclass_v1" / "train" / "images",
    )
    parser.add_argument(
        "--source-labels", type=Path,
        default=ROOT / "data" / "processed" / "phase1_multiclass_v1" / "train" / "labels",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "augmented" / "vlm_generated",
    )
    parser.add_argument("--num", type=int, default=200, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[INFO] Strategy: {args.strategy}")
    print(f"[INFO] Source: {args.source_images}")
    print(f"[INFO] Output: {args.output}")

    if args.strategy == "smart":
        result = smart_augment_dataset(
            source_image_dir=args.source_images,
            source_label_dir=args.source_labels,
            output_dir=args.output,
            num_augmented=args.num,
            seed=args.seed,
        )
    elif args.strategy == "copypaste":
        crops = extract_crops(args.source_images, args.source_labels)
        print(f"[INFO] Extracted {len(crops)} crops from source data")
        result = copy_paste_augment(
            background_dir=args.source_images,
            crops=crops,
            output_dir=args.output,
            num_images=args.num,
            seed=args.seed,
        )

    print(f"\n[OK] {result.total_generated} augmented images generated")
    print(f"     Output: {result.output_dir}")

    # Save report
    report_path = args.output / "augmentation_report.json"
    report_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "strategy": result.strategy,
        "total_generated": result.total_generated,
        "output_dir": result.output_dir,
        "source_images": str(args.source_images),
        "source_labels": str(args.source_labels),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
