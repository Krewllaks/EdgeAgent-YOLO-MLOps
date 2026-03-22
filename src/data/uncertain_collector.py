"""
Uncertain Frame Collector — Belirsiz kareleri biriktirme modulu.

Uretim bandinda YOLO'nun dusuk confidence ile tespit ettigi
kareleri biriktirir. Bu kareler sonra:
1. Copy-paste augmentation ile cogaltilir
2. Stable Diffusion ile varyasyonlari uretilir
3. Model yeniden egitilir (continuous training)

Hocanin vizyonu: "Gunduz uretim, gece iyilestirme" dongusu.
"""

import json
import shutil
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNCERTAIN_DIR = ROOT / "data" / "uncertain"
METADATA_FILE = "metadata.jsonl"


@dataclass
class UncertainFrame:
    """A frame that YOLO was uncertain about."""
    timestamp: str
    image_path: str
    min_confidence: float
    max_confidence: float
    avg_confidence: float
    detection_count: int
    detections: list  # [{class_id, class_name, confidence, bbox}]
    trigger_reason: str  # "low_confidence" | "no_detection" | "conflict"
    collected: bool = True


class UncertainCollector:
    """Collects uncertain frames for later augmentation and retraining."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_UNCERTAIN_DIR,
        confidence_threshold: float = 0.40,
        max_stored: int = 1000,
    ):
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.max_stored = max_stored

        self.images_dir = output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = output_dir / METADATA_FILE

    def should_collect(self, detections: list, min_conf: float = None) -> tuple:
        """Check if this frame should be collected based on detection confidence.

        Args:
            detections: List of dicts with {class_id, confidence, bbox}
            min_conf: Override confidence threshold

        Returns:
            (should_collect: bool, reason: str)
        """
        threshold = min_conf or self.confidence_threshold

        if not detections:
            return True, "no_detection"

        confidences = [d.get("confidence", 0) for d in detections]
        min_c = min(confidences)

        if min_c < threshold:
            return True, "low_confidence"

        return False, "ok"

    def collect_frame(
        self,
        image,
        detections: list,
        reason: str = "low_confidence",
    ) -> Optional[UncertainFrame]:
        """Save an uncertain frame for later processing.

        Args:
            image: Path to source image OR numpy array (BGR)
            detections: YOLO detections [{class_id, confidence, bbox, class_name}]
            reason: Why this frame was collected

        Returns:
            UncertainFrame record, or None if storage is full
        """
        import cv2

        # Check storage limit
        existing = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        if len(existing) >= self.max_stored:
            # Remove oldest
            oldest = min(existing, key=lambda p: p.stat().st_mtime)
            oldest.unlink()

        # Save image — accept both Path and numpy array
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if isinstance(image, (str, Path)):
            dest_name = f"uncertain_{timestamp}{Path(image).suffix}"
            dest_path = self.images_dir / dest_name
            shutil.copy2(image, dest_path)
        elif isinstance(image, np.ndarray):
            dest_name = f"uncertain_{timestamp}.jpg"
            dest_path = self.images_dir / dest_name
            cv2.imwrite(str(dest_path), image)
        else:
            return None

        # Compute stats
        confidences = [d.get("confidence", 0) for d in detections] if detections else [0]

        frame = UncertainFrame(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            image_path=str(dest_path),
            min_confidence=float(min(confidences)),
            max_confidence=float(max(confidences)),
            avg_confidence=float(np.mean(confidences)),
            detection_count=len(detections),
            detections=detections,
            trigger_reason=reason,
        )

        # Append to metadata
        with open(self.metadata_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(frame), ensure_ascii=False) + "\n")

        return frame

    def get_collected_count(self) -> int:
        """Return number of collected uncertain frames."""
        images = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        return len(images)

    def get_summary(self) -> dict:
        """Get summary statistics of collected uncertain frames."""
        if not self.metadata_path.exists():
            return {"total": 0, "reasons": {}, "avg_confidence": 0}

        entries = []
        for line in self.metadata_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not entries:
            return {"total": 0, "reasons": {}, "avg_confidence": 0}

        reasons = {}
        confidences = []
        for e in entries:
            r = e.get("trigger_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
            confidences.append(e.get("avg_confidence", 0))

        return {
            "total": len(entries),
            "reasons": reasons,
            "avg_confidence": float(np.mean(confidences)) if confidences else 0,
            "oldest": entries[0].get("timestamp", ""),
            "newest": entries[-1].get("timestamp", ""),
        }

    def clear(self) -> int:
        """Clear all collected uncertain frames. Returns count removed."""
        count = 0
        for f in self.images_dir.iterdir():
            if f.is_file():
                f.unlink()
                count += 1
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        return count

    def get_image_paths(self) -> list:
        """Return all collected image paths."""
        return sorted(
            self.images_dir.glob("*"),
            key=lambda p: p.stat().st_mtime,
        )

    def get_unlabeled_images(self) -> list:
        """Return images that don't have a VLM pseudo-label yet."""
        labels_dir = self.output_dir / "labels"
        images = self.get_image_paths()
        unlabeled = []
        for img in images:
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                label_path = labels_dir / f"{img.stem}.txt"
                if not label_path.exists():
                    unlabeled.append(img)
        return unlabeled

    def save_pseudo_label(self, image_path: Path, label_lines: list) -> Path:
        """Save VLM-generated pseudo-label for an uncertain frame.

        Args:
            image_path: Path to the uncertain image
            label_lines: YOLO format lines ["class_id cx cy w h", ...]

        Returns:
            Path to saved label file
        """
        labels_dir = self.output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_path = labels_dir / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(label_lines), encoding="utf-8")
        return label_path

    def get_labeled_pairs(self) -> list:
        """Return (image_path, label_path) pairs for CT-ready data."""
        labels_dir = self.output_dir / "labels"
        if not labels_dir.exists():
            return []
        pairs = []
        for label_path in sorted(labels_dir.glob("*.txt")):
            if label_path.stat().st_size == 0:
                continue
            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = self.images_dir / f"{label_path.stem}{ext}"
                if img_path.exists():
                    pairs.append((img_path, label_path))
                    break
        return pairs
