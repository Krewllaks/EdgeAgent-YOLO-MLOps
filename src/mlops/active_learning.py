"""Active Learning Pipeline - Operator feedback to model retraining.

Collects operator feedback from the dashboard, prepares retraining
datasets, and manages the 2-week retraining cycle.

Usage:
    from src.mlops.active_learning import ActiveLearningPipeline

    pipeline = ActiveLearningPipeline()
    summary = pipeline.collect_feedback()
    if pipeline.should_retrain(summary):
        manifest = pipeline.prepare_retrain_set(Path("data/retrain"))
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEEDBACK_PATH = ROOT / "data" / "feedback" / "feedback_log.jsonl"
DEFAULT_RETRAIN_LOG = ROOT / "data" / "feedback" / "retrain_history.jsonl"


@dataclass
class FeedbackSummary:
    """Summary of collected operator feedback."""

    total: int
    correct: int
    incorrect: int
    partial: int
    accuracy: float
    by_image: dict  # image_name -> label
    entries: list[dict]


@dataclass
class RetrainManifest:
    """Manifest for a retraining dataset."""

    output_dir: Path
    total_images: int
    incorrect_images: int
    data_yaml_path: Path
    timestamp: str


class ActiveLearningPipeline:
    """Manages operator feedback collection and retraining triggers."""

    def __init__(
        self,
        feedback_path: Path = DEFAULT_FEEDBACK_PATH,
        retrain_log_path: Path = DEFAULT_RETRAIN_LOG,
        retrain_cycle_days: int = 14,
        min_feedback_samples: int = 50,
    ):
        self.feedback_path = feedback_path
        self.retrain_log_path = retrain_log_path
        self.retrain_cycle_days = retrain_cycle_days
        self.min_feedback_samples = min_feedback_samples

    def collect_feedback(self) -> FeedbackSummary:
        """Parse feedback log and return summary.

        Supports both legacy format (label only) and new granular format
        (per-detection correct/incorrect with bbox data).
        """
        if not self.feedback_path.exists():
            return FeedbackSummary(
                total=0, correct=0, incorrect=0, partial=0,
                accuracy=1.0, by_image={}, entries=[],
            )

        entries = []
        for line in self.feedback_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                entries.append(json.loads(line))

        correct = sum(1 for e in entries if e.get("label") == "correct")
        incorrect = sum(1 for e in entries if e.get("label") == "incorrect")
        partial = sum(1 for e in entries if e.get("label") == "partial")
        total = len(entries)
        # For accuracy: partial counts as half-correct
        accuracy = (correct + partial * 0.5) / max(1, total)

        by_image = {}
        for e in entries:
            img = e.get("image", "unknown")
            by_image[img] = e.get("label", "unknown")

        return FeedbackSummary(
            total=total,
            correct=correct,
            incorrect=incorrect,
            partial=partial,
            accuracy=accuracy,
            by_image=by_image,
            entries=entries,
        )

    def get_last_retrain_date(self) -> Optional[datetime]:
        """Get timestamp of last retraining run."""
        if not self.retrain_log_path.exists():
            return None

        last_line = ""
        for line in self.retrain_log_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                last_line = line

        if last_line:
            entry = json.loads(last_line)
            return datetime.fromisoformat(entry.get("timestamp", ""))
        return None

    def should_retrain(self, summary: Optional[FeedbackSummary] = None) -> bool:
        """Check if retraining should be triggered.

        Conditions (ALL must be met):
        1. At least min_feedback_samples incorrect+partial feedback entries
        2. At least retrain_cycle_days since last retrain
        """
        if summary is None:
            summary = self.collect_feedback()

        # Condition 1: Enough samples with corrections needed
        needs_correction = summary.incorrect + summary.partial
        if needs_correction < self.min_feedback_samples:
            logger.info(
                "Not enough corrective feedback: %d < %d",
                needs_correction, self.min_feedback_samples,
            )
            return False

        # Condition 2: Time since last retrain
        last_retrain = self.get_last_retrain_date()
        if last_retrain is not None:
            days_since = (datetime.now() - last_retrain).days
            if days_since < self.retrain_cycle_days:
                logger.info(
                    "Too soon since last retrain: %d days < %d",
                    days_since, self.retrain_cycle_days,
                )
                return False

        return True

    @staticmethod
    def _generate_corrected_labels(
        entry: dict, img_width: int, img_height: int,
    ) -> str:
        """Generate corrected YOLO-format labels from per-detection feedback.

        For 'partial' feedback: keep only detections marked correct.
        For 'incorrect' with detections: keep only correct ones (if any).
        Converts bbox from xyxy (pixel) to YOLO normalized xywh format.

        Returns:
            YOLO label text (one line per correct detection).
        """
        detections = entry.get("detections", [])
        lines = []
        for det in detections:
            if not det.get("correct", True):
                continue  # skip operator-rejected detections
            cls_id = det["class_id"]
            bbox = det["bbox"]  # [x1, y1, x2, y2] in pixels
            # Convert xyxy -> xywh normalized
            x_center = ((bbox[0] + bbox[2]) / 2.0) / img_width
            y_center = ((bbox[1] + bbox[3]) / 2.0) / img_height
            w = (bbox[2] - bbox[0]) / img_width
            h = (bbox[3] - bbox[1]) / img_height
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        return "\n".join(lines)

    def prepare_retrain_set(
        self,
        output_dir: Path,
        source_image_dir: Optional[Path] = None,
    ) -> RetrainManifest:
        """Prepare a retraining dataset from operator feedback.

        Supports two feedback types:
        - Legacy ('incorrect'): copies image + original labels for manual correction
        - Granular ('partial'/'incorrect' with detections): generates corrected
          YOLO labels from per-detection feedback (keeps only correct detections)

        Args:
            output_dir: Where to create the retrain dataset.
            source_image_dir: Directory containing source images.

        Returns:
            RetrainManifest with dataset info.
        """
        if source_image_dir is None:
            source_image_dir = ROOT / "data" / "processed" / "phase1_multiclass_v1" / "test" / "images"

        summary = self.collect_feedback()
        # Collect entries that need correction (incorrect or partial)
        correction_entries = [
            e for e in summary.entries
            if e.get("label") in ("incorrect", "partial")
        ]

        # Create output structure
        retrain_images = output_dir / "images" / "train"
        retrain_labels = output_dir / "labels" / "train"
        retrain_images.mkdir(parents=True, exist_ok=True)
        retrain_labels.mkdir(parents=True, exist_ok=True)

        copied = 0
        for entry in correction_entries:
            img_name = entry.get("image", "")
            src_img = source_image_dir / img_name
            # Also check in uploaded tmp images
            if not src_img.exists():
                tmp_img = ROOT / "reports" / "generated" / img_name
                if tmp_img.exists():
                    src_img = tmp_img

            if not src_img.exists():
                logger.warning("Image not found: %s", img_name)
                continue

            # Copy image
            shutil.copy2(src_img, retrain_images / img_name)

            # Generate corrected label
            label_name = Path(img_name).stem + ".txt"
            has_granular = bool(entry.get("detections"))

            if has_granular:
                # New granular feedback: generate corrected labels from operator markings
                from PIL import Image as PILImage
                try:
                    img = PILImage.open(src_img)
                    w, h = img.size
                except Exception:
                    w, h = 640, 640  # fallback
                corrected_labels = self._generate_corrected_labels(entry, w, h)
                (retrain_labels / label_name).write_text(corrected_labels, encoding="utf-8")
                logger.info(
                    "Generated corrected labels for %s: %d/%d detections kept",
                    img_name,
                    sum(1 for d in entry["detections"] if d.get("correct", True)),
                    len(entry["detections"]),
                )
            else:
                # Legacy feedback: copy original label if exists
                src_label = source_image_dir.parent.parent / "labels" / "train" / label_name
                if not src_label.exists():
                    src_label = source_image_dir.parent.parent / "labels" / "test" / label_name
                if src_label.exists():
                    shutil.copy2(src_label, retrain_labels / label_name)

            copied += 1

        # Create data.yaml
        data_yaml = output_dir / "data.yaml"
        yaml_content = (
            f"path: {output_dir.resolve()}\n"
            f"train: images/train\n"
            f"val: images/train\n"
            f"nc: 3\n"
            f"names:\n"
            f"  0: screw\n"
            f"  1: missing_screw\n"
            f"  2: missing_component\n"
        )
        data_yaml.write_text(yaml_content, encoding="utf-8")

        manifest = RetrainManifest(
            output_dir=output_dir,
            total_images=copied,
            incorrect_images=len(correction_entries),
            data_yaml_path=data_yaml,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

        logger.info(
            "Retrain set prepared: %d images (%d granular, %d legacy) in %s",
            copied,
            sum(1 for e in correction_entries if e.get("detections")),
            sum(1 for e in correction_entries if not e.get("detections")),
            output_dir,
        )
        return manifest

    def log_retrain(self, manifest: RetrainManifest, metrics: Optional[dict] = None) -> None:
        """Log a completed retraining run."""
        self.retrain_log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "images": manifest.total_images,
            "output_dir": str(manifest.output_dir),
            "metrics": metrics or {},
        }
        with self.retrain_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=== Active Learning Pipeline Self-Test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create fake feedback with both legacy and granular entries
        feedback_file = tmpdir / "feedback.jsonl"
        entries = []
        for i in range(40):
            # Legacy format: correct
            entries.append({
                "timestamp": datetime.now().isoformat(),
                "image": f"test_{i}.jpg",
                "label": "correct",
                "detection_count": 3,
            })
        for i in range(40, 55):
            # Legacy format: incorrect
            entries.append({
                "timestamp": datetime.now().isoformat(),
                "image": f"test_{i}.jpg",
                "label": "incorrect",
                "detection_count": 3,
            })
        for i in range(55, 70):
            # Granular format: partial
            entries.append({
                "timestamp": datetime.now().isoformat(),
                "image": f"test_{i}.jpg",
                "label": "partial",
                "detection_count": 4,
                "correct_count": 3,
                "incorrect_count": 1,
                "detections": [
                    {"idx": 0, "class_id": 0, "class_name": "screw",
                     "confidence": 0.95, "bbox": [10, 20, 50, 60], "correct": True},
                    {"idx": 1, "class_id": 0, "class_name": "screw",
                     "confidence": 0.88, "bbox": [100, 200, 150, 260], "correct": True},
                    {"idx": 2, "class_id": 1, "class_name": "missing_screw",
                     "confidence": 0.75, "bbox": [200, 300, 250, 360], "correct": True},
                    {"idx": 3, "class_id": 0, "class_name": "screw",
                     "confidence": 0.32, "bbox": [400, 400, 420, 420], "correct": False},
                ],
            })
        feedback_file.write_text(
            "\n".join(json.dumps(e) for e in entries),
            encoding="utf-8",
        )

        pipeline = ActiveLearningPipeline(
            feedback_path=feedback_file,
            retrain_log_path=tmpdir / "retrain_history.jsonl",
            retrain_cycle_days=14,
            min_feedback_samples=10,
        )

        # Test 1: Collect feedback
        summary = pipeline.collect_feedback()
        assert summary.total == 70
        assert summary.correct == 40
        assert summary.incorrect == 15
        assert summary.partial == 15
        print(f"Test 1 PASS: Collected {summary.total} entries, "
              f"{summary.incorrect} incorrect, {summary.partial} partial, "
              f"accuracy={summary.accuracy:.0%}")

        # Test 2: Should retrain (enough samples: 15 incorrect + 15 partial = 30 >= 10)
        assert pipeline.should_retrain(summary)
        print("Test 2 PASS: should_retrain=True (enough corrective samples)")

        # Test 3: Generate corrected labels
        test_entry = entries[55]  # a partial entry
        labels_text = pipeline._generate_corrected_labels(test_entry, 640, 640)
        label_lines = [l for l in labels_text.strip().split("\n") if l.strip()]
        assert len(label_lines) == 3, f"Expected 3 correct dets, got {len(label_lines)}"
        print(f"Test 3 PASS: Generated corrected labels (3/4 kept)")

        # Test 4: Log retrain and check cycle
        manifest = RetrainManifest(
            output_dir=tmpdir / "retrain",
            total_images=30,
            incorrect_images=30,
            data_yaml_path=tmpdir / "retrain" / "data.yaml",
            timestamp=datetime.now().isoformat(),
        )
        pipeline.log_retrain(manifest, {"mAP50": 0.95})
        assert not pipeline.should_retrain(summary)  # Too soon
        print("Test 4 PASS: should_retrain=False after logging (14-day cycle)")

        # Test 5: Empty feedback
        empty_pipeline = ActiveLearningPipeline(
            feedback_path=tmpdir / "nonexistent.jsonl",
        )
        empty_summary = empty_pipeline.collect_feedback()
        assert empty_summary.total == 0
        assert empty_summary.partial == 0
        assert empty_summary.accuracy == 1.0
        print("Test 5 PASS: Empty feedback handled correctly")

    print("\n=== All 5 tests passed ===")
