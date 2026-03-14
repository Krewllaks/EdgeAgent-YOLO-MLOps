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
        """Parse feedback log and return summary."""
        if not self.feedback_path.exists():
            return FeedbackSummary(
                total=0, correct=0, incorrect=0, accuracy=1.0,
                by_image={}, entries=[],
            )

        entries = []
        for line in self.feedback_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                entries.append(json.loads(line))

        correct = sum(1 for e in entries if e.get("label") == "correct")
        incorrect = sum(1 for e in entries if e.get("label") == "incorrect")
        total = len(entries)
        accuracy = correct / max(1, total)

        by_image = {}
        for e in entries:
            img = e.get("image", "unknown")
            by_image[img] = e.get("label", "unknown")

        return FeedbackSummary(
            total=total,
            correct=correct,
            incorrect=incorrect,
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
        1. At least min_feedback_samples incorrect feedback entries
        2. At least retrain_cycle_days since last retrain
        """
        if summary is None:
            summary = self.collect_feedback()

        # Condition 1: Enough incorrect samples
        if summary.incorrect < self.min_feedback_samples:
            logger.info(
                "Not enough incorrect feedback: %d < %d",
                summary.incorrect, self.min_feedback_samples,
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

    def prepare_retrain_set(
        self,
        output_dir: Path,
        source_image_dir: Optional[Path] = None,
    ) -> RetrainManifest:
        """Prepare a retraining dataset from incorrect feedback entries.

        Collects images that operators marked as 'incorrect' and
        creates a YOLO-format dataset structure for fine-tuning.

        Args:
            output_dir: Where to create the retrain dataset.
            source_image_dir: Directory containing source images
                (defaults to test images directory).

        Returns:
            RetrainManifest with dataset info.
        """
        if source_image_dir is None:
            source_image_dir = ROOT / "data" / "processed" / "phase1_multiclass_v1" / "test" / "images"

        summary = self.collect_feedback()
        incorrect_images = [
            e.get("image", "") for e in summary.entries
            if e.get("label") == "incorrect"
        ]

        # Create output structure
        retrain_images = output_dir / "images" / "train"
        retrain_labels = output_dir / "labels" / "train"
        retrain_images.mkdir(parents=True, exist_ok=True)
        retrain_labels.mkdir(parents=True, exist_ok=True)

        copied = 0
        for img_name in incorrect_images:
            src_img = source_image_dir / img_name
            if src_img.exists():
                shutil.copy2(src_img, retrain_images / img_name)
                # Copy corresponding label if exists
                label_name = src_img.stem + ".txt"
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
            f"val: images/train\n"  # Use same set for fine-tuning
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
            incorrect_images=len(incorrect_images),
            data_yaml_path=data_yaml,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

        logger.info(
            "Retrain set prepared: %d images in %s",
            copied, output_dir,
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

        # Create fake feedback
        feedback_file = tmpdir / "feedback.jsonl"
        entries = []
        for i in range(60):
            label = "incorrect" if i % 3 == 0 else "correct"
            entries.append({
                "timestamp": datetime.now().isoformat(),
                "image": f"test_{i}.jpg",
                "label": label,
                "detection_count": 3,
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
        assert summary.total == 60
        assert summary.incorrect == 20  # every 3rd
        assert summary.correct == 40
        print(f"Test 1 PASS: Collected {summary.total} entries, "
              f"{summary.incorrect} incorrect, accuracy={summary.accuracy:.0%}")

        # Test 2: Should retrain (enough samples, no prior retrain)
        assert pipeline.should_retrain(summary)
        print("Test 2 PASS: should_retrain=True (enough samples, no prior)")

        # Test 3: Log retrain and check cycle
        manifest = RetrainManifest(
            output_dir=tmpdir / "retrain",
            total_images=20,
            incorrect_images=20,
            data_yaml_path=tmpdir / "retrain" / "data.yaml",
            timestamp=datetime.now().isoformat(),
        )
        pipeline.log_retrain(manifest, {"mAP50": 0.95})
        assert not pipeline.should_retrain(summary)  # Too soon
        print("Test 3 PASS: should_retrain=False after logging (14-day cycle)")

        # Test 4: Not enough samples
        pipeline2 = ActiveLearningPipeline(
            feedback_path=feedback_file,
            retrain_log_path=tmpdir / "retrain2.jsonl",
            min_feedback_samples=100,  # Require more than we have
        )
        assert not pipeline2.should_retrain(summary)
        print("Test 4 PASS: should_retrain=False (not enough samples)")

        # Test 5: Empty feedback
        empty_pipeline = ActiveLearningPipeline(
            feedback_path=tmpdir / "nonexistent.jsonl",
        )
        empty_summary = empty_pipeline.collect_feedback()
        assert empty_summary.total == 0
        assert empty_summary.accuracy == 1.0
        print("Test 5 PASS: Empty feedback handled correctly")

    print("\n=== All 5 tests passed ===")
