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

from src.common.constants import IMAGE_EXTS


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


# Domain constraints: maximum possible detections per class
# Based on the physical hardware being inspected
MAX_PER_CLASS = {
    0: 4,   # screw: max 4
    1: 4,   # missing_screw: max 4
    2: 2,   # missing_component: max 2
}
MAX_TOTAL_DETECTIONS = 10  # absolute upper bound


def _count_per_class(boxes) -> dict[int, int]:
    """Count detections per class from YOLO boxes."""
    counts: dict[int, int] = {}
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        counts[cls_id] = counts.get(cls_id, 0) + 1
    return counts


def _is_valid_detection(boxes) -> bool:
    """Check if detection counts are physically plausible.

    Returns False if any class exceeds its max count (false positive explosion).
    """
    if len(boxes) > MAX_TOTAL_DETECTIONS:
        return False
    counts = _count_per_class(boxes)
    for cls_id, count in counts.items():
        if count > MAX_PER_CLASS.get(cls_id, 4):
            return False
    return True


def _smart_score(boxes) -> float:
    """Score detections with domain-aware penalty.

    Base score: sum of confidences (rewards high-confidence detections).
    Penalty: if any class exceeds max count, score = 0 (invalid).
    Bonus: higher avg confidence = better (fewer false positives).
    """
    n = len(boxes)
    if n == 0:
        return 0.0

    # Hard reject if physically impossible
    if not _is_valid_detection(boxes):
        return -1.0

    # Score = avg_confidence * valid_detection_count
    # This rewards finding real objects with high confidence
    # rather than finding many low-confidence false positives
    avg_conf = float(boxes.conf.mean())
    min_conf = float(boxes.conf.min())

    # Penalize if any detection is very low confidence (likely FP)
    low_conf_penalty = max(0.0, 1.0 - max(0.0, 0.40 - min_conf) * 2)

    return n * avg_conf * low_conf_penalty


def auto_tune_fast(
    img: np.ndarray,
    model,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    class_limits: dict[int, int] | None = None,
) -> dict:
    """Find optimal Canny parameters by maximizing YOLO detection quality.

    Coarse-to-fine search with domain constraints:
    - Rejects parameter sets that produce physically impossible detections
      (e.g., >4 screws when hardware has max 4 screw positions)
    - Penalizes low-confidence detections (likely false positives from edges)
    - Prefers the original image if enhancement doesn't improve quality

    Args:
        img: BGR image (OpenCV format)
        model: YOLO model instance
        imgsz: YOLO input size
        conf: YOLO confidence threshold
        iou: YOLO IoU threshold for NMS
        class_limits: Override max detections per class {cls_id: max_count}
    """
    import tempfile
    import os

    # Apply custom class limits if provided
    if class_limits:
        for cls_id, limit in class_limits.items():
            MAX_PER_CLASS[cls_id] = limit

    tmp_dir = tempfile.mkdtemp()
    enh_path = os.path.join(tmp_dir, "enh.jpg")
    orig_path = os.path.join(tmp_dir, "orig.jpg")
    cv2.imwrite(orig_path, img)

    def _evaluate(image_path):
        r = model.predict(image_path, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        boxes = r[0].boxes
        return _smart_score(boxes), len(boxes), boxes

    def _score_params(alpha, cl, ch):
        enhanced = enhance_single(img, alpha, cl, ch)
        cv2.imwrite(enh_path, enhanced)
        score, n, boxes = _evaluate(enh_path)
        avg_c = float(boxes.conf.mean()) if n > 0 else 0.0
        valid = _is_valid_detection(boxes)
        per_class = _count_per_class(boxes)
        return score, n, avg_c, valid, per_class

    # Baseline (original)
    orig_score, orig_n, orig_boxes = _evaluate(orig_path)
    orig_avg = float(orig_boxes.conf.mean()) if orig_n > 0 else 0.0
    orig_valid = _is_valid_detection(orig_boxes)
    orig_per_class = _count_per_class(orig_boxes)

    best_score = orig_score
    best_params = (0.7, 50, 150)
    best_n, best_avg = orig_n, orig_avg
    best_valid = orig_valid
    best_per_class = orig_per_class

    # Coarse search
    for alpha in [0.5, 0.7, 0.85]:
        for cl in [30, 60, 100]:
            for ch in [100, 170, 250]:
                if ch <= cl:
                    continue
                s, n, a, v, pc = _score_params(alpha, cl, ch)
                if s > best_score:
                    best_score = s
                    best_params = (alpha, cl, ch)
                    best_n, best_avg = n, a
                    best_valid = v
                    best_per_class = pc

    # Fine search around best
    ba, bcl, bch = best_params
    for alpha in [max(0.3, ba - 0.1), ba, min(1.0, ba + 0.1)]:
        for cl in [max(10, bcl - 15), bcl, bcl + 15]:
            for ch in [max(50, bch - 25), bch, bch + 25]:
                if ch <= cl:
                    continue
                s, n, a, v, pc = _score_params(alpha, cl, ch)
                if s > best_score:
                    best_score = s
                    best_params = (alpha, cl, ch)
                    best_n, best_avg = n, a
                    best_valid = v
                    best_per_class = pc

    # Cleanup
    try:
        os.remove(orig_path)
        os.remove(enh_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    is_original = best_score <= orig_score
    return {
        "alpha": best_params[0],
        "canny_low": best_params[1],
        "canny_high": best_params[2],
        "score": best_score,
        "det_count": best_n,
        "avg_conf": best_avg,
        "original_score": orig_score,
        "is_original": is_original,
        "valid": best_valid,
        "per_class": best_per_class,
        "orig_per_class": orig_per_class,
    }


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
