"""Canny Edge Enhancement for metal surface reflection suppression.

Blends Canny edge maps with original images to help YOLO detect
features on highly reflective metal surfaces.

Usage:
    # Single image preview
    python src/data/edge_enhancer.py --image path/to/img.jpg --preview

    # Batch enhance dataset
    python src/data/edge_enhancer.py --input-dir data/processed/phase1_v2/train/images \
                                     --output-dir data/processed/phase1_v2_edge/train/images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def enhance_single(
    img: np.ndarray,
    alpha: float = 0.7,
    canny_low: int = 50,
    canny_high: int = 150,
) -> np.ndarray:
    """Blend Canny edge map with original image.

    Args:
        img: BGR image (OpenCV format)
        alpha: Weight for original image (1-alpha for edges)
        canny_low: Canny lower threshold
        canny_high: Canny upper threshold

    Returns:
        Blended BGR image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(img, alpha, edges_bgr, 1.0 - alpha, 0)
    return blended


def enhance_from_path(
    img_path: str | Path,
    alpha: float = 0.7,
    canny_low: int = 50,
    canny_high: int = 150,
) -> np.ndarray:
    """Load image from path and apply edge enhancement."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return enhance_single(img, alpha, canny_low, canny_high)


def preview_enhancement(
    img_path: str | Path,
    alpha: float = 0.7,
    canny_low: int = 50,
    canny_high: int = 150,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (original, edges_only, blended) for dashboard display.

    All returned images are in RGB format for matplotlib/streamlit.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    blended = enhance_single(img, alpha, canny_low, canny_high)

    # Convert BGR -> RGB for display
    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    return original_rgb, edges_rgb, blended_rgb


def enhance_dataset(
    input_dir: Path,
    output_dir: Path,
    alpha: float = 0.7,
    canny_low: int = 50,
    canny_high: int = 150,
) -> dict:
    """Batch-enhance all images in a directory.

    Returns:
        Stats dict with processed/failed counts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"processed": 0, "failed": 0, "skipped": 0}

    images = [p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    total = len(images)

    for i, img_path in enumerate(images):
        dst = output_dir / img_path.name
        if dst.exists():
            stats["skipped"] += 1
            continue
        try:
            enhanced = enhance_from_path(img_path, alpha, canny_low, canny_high)
            cv2.imwrite(str(dst), enhanced)
            stats["processed"] += 1
        except Exception as e:
            print(f"[WARN] Failed to enhance {img_path.name}: {e}")
            stats["failed"] += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] processed...")

    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canny Edge Enhancement for YOLO preprocessing")
    p.add_argument("--image", type=Path, help="Single image to enhance")
    p.add_argument("--input-dir", type=Path, help="Directory of images to batch enhance")
    p.add_argument("--output-dir", type=Path, help="Output directory for enhanced images")
    p.add_argument("--alpha", type=float, default=0.7, help="Original image weight (default: 0.7)")
    p.add_argument("--canny-low", type=int, default=50, help="Canny low threshold (default: 50)")
    p.add_argument("--canny-high", type=int, default=150, help="Canny high threshold (default: 150)")
    p.add_argument("--preview", action="store_true", help="Show preview window (single image mode)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.image:
        enhanced = enhance_from_path(args.image, args.alpha, args.canny_low, args.canny_high)
        if args.preview:
            cv2.imshow("Original", cv2.imread(str(args.image)))
            cv2.imshow("Enhanced", enhanced)
            print("[INFO] Press any key to close preview windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            out_path = args.image.parent / f"{args.image.stem}_enhanced{args.image.suffix}"
            cv2.imwrite(str(out_path), enhanced)
            print(f"[OK] Enhanced image saved: {out_path}")

    elif args.input_dir and args.output_dir:
        if not args.input_dir.exists():
            sys.exit(f"[ERR] Input dir not found: {args.input_dir}")
        print(f"[INFO] Enhancing images from {args.input_dir}")
        stats = enhance_dataset(
            args.input_dir, args.output_dir,
            args.alpha, args.canny_low, args.canny_high,
        )
        print(f"[OK] Batch enhancement complete: {stats}")

    else:
        print("[ERR] Provide --image for single file or --input-dir + --output-dir for batch")
        sys.exit(1)


if __name__ == "__main__":
    main()
