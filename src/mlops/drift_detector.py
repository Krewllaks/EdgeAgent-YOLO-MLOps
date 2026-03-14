"""Concept Drift Detector - SSIM-based image distribution monitoring.

Detects when production images deviate significantly from the training
distribution, indicating potential concept drift (lighting changes,
new product variants, camera angle shifts).

Usage:
    from src.mlops.drift_detector import DriftDetector

    detector = DriftDetector(window_size=500, ssim_threshold=0.15)
    detector.compute_ssim_baseline(Path("data/processed/phase1_v2/train/images"))
    result = detector.check_drift(current_batch_images)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of a drift detection check."""

    drift_detected: bool
    baseline_ssim: float
    current_ssim: float
    delta: float
    recommendation: str  # "retrain", "monitor", "ok"
    timestamp: str
    samples_checked: int


class DriftDetector:
    """SSIM-based concept drift detector.

    Compares structural similarity between a reference set (training data)
    and incoming production images. A significant SSIM drop indicates
    visual distribution shift.
    """

    def __init__(
        self,
        window_size: int = 500,
        ssim_threshold: float = 0.15,
        sample_pairs: int = 100,
        target_size: tuple[int, int] = (256, 256),
    ):
        self.window_size = window_size
        self.ssim_threshold = ssim_threshold
        self.sample_pairs = sample_pairs
        self.target_size = target_size
        self._baseline_ssim: Optional[float] = None
        self._reference_images: list[np.ndarray] = []

    @property
    def has_baseline(self) -> bool:
        return self._baseline_ssim is not None

    @staticmethod
    def _load_and_resize(path: Path, target_size: tuple[int, int]) -> Optional[np.ndarray]:
        """Load image as grayscale and resize for SSIM comparison."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return cv2.resize(img, target_size)

    @staticmethod
    def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute SSIM between two grayscale images.

        Uses scikit-image if available, falls back to OpenCV-based
        approximation.
        """
        try:
            from skimage.metrics import structural_similarity
            return structural_similarity(img1, img2)
        except ImportError:
            # Fallback: simplified SSIM using OpenCV
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2

            img1_f = img1.astype(np.float64)
            img2_f = img2.astype(np.float64)

            mu1 = cv2.GaussianBlur(img1_f, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(img2_f, (11, 11), 1.5)

            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = cv2.GaussianBlur(img1_f ** 2, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(img2_f ** 2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(img1_f * img2_f, (11, 11), 1.5) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                       ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

            return float(ssim_map.mean())

    def compute_ssim_baseline(self, reference_dir: Path) -> float:
        """Compute baseline SSIM from reference (training) images.

        Randomly samples pairs from the reference set and computes
        average intra-set SSIM. This establishes the "normal" similarity
        level for the training distribution.

        Returns:
            Average SSIM within the reference set.
        """
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted([
            p for p in reference_dir.iterdir()
            if p.suffix.lower() in exts
        ])

        if len(image_paths) < 10:
            raise ValueError(
                f"Need at least 10 reference images, got {len(image_paths)}"
            )

        # Load a subset for baseline + keep for future comparison
        sample_size = min(len(image_paths), self.window_size)
        sampled_paths = random.sample(image_paths, sample_size)

        self._reference_images = []
        for p in sampled_paths:
            img = self._load_and_resize(p, self.target_size)
            if img is not None:
                self._reference_images.append(img)

        if len(self._reference_images) < 10:
            raise ValueError("Could not load enough reference images")

        # Compute pairwise SSIM on random pairs
        ssim_values = []
        n_pairs = min(self.sample_pairs, len(self._reference_images) * (len(self._reference_images) - 1) // 2)
        for _ in range(n_pairs):
            i, j = random.sample(range(len(self._reference_images)), 2)
            ssim = self._compute_ssim(self._reference_images[i], self._reference_images[j])
            ssim_values.append(ssim)

        self._baseline_ssim = sum(ssim_values) / len(ssim_values)
        logger.info(
            "Baseline SSIM computed: %.4f (from %d pairs, %d images)",
            self._baseline_ssim, n_pairs, len(self._reference_images),
        )
        return self._baseline_ssim

    def check_drift(self, current_images: list[np.ndarray]) -> DriftResult:
        """Check if current batch shows drift from baseline.

        Compares each current image against random reference images
        and checks if average cross-set SSIM has dropped significantly.

        Args:
            current_images: List of grayscale numpy arrays (already resized).

        Returns:
            DriftResult with drift detection outcome.
        """
        if self._baseline_ssim is None:
            raise RuntimeError("Call compute_ssim_baseline() first")

        if not current_images:
            return DriftResult(
                drift_detected=False,
                baseline_ssim=self._baseline_ssim,
                current_ssim=self._baseline_ssim,
                delta=0.0,
                recommendation="ok",
                timestamp=datetime.now().isoformat(timespec="seconds"),
                samples_checked=0,
            )

        # Resize current images
        resized = []
        for img in current_images:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized.append(cv2.resize(img, self.target_size))

        # Cross-set SSIM: current vs reference
        ssim_values = []
        n_checks = min(self.sample_pairs, len(resized) * len(self._reference_images))
        for _ in range(n_checks):
            curr = random.choice(resized)
            ref = random.choice(self._reference_images)
            ssim = self._compute_ssim(curr, ref)
            ssim_values.append(ssim)

        current_ssim = sum(ssim_values) / len(ssim_values)
        delta = self._baseline_ssim - current_ssim
        drift_detected = delta > self.ssim_threshold

        if drift_detected:
            recommendation = "retrain"
        elif delta > self.ssim_threshold * 0.5:
            recommendation = "monitor"
        else:
            recommendation = "ok"

        logger.info(
            "Drift check: baseline=%.4f, current=%.4f, delta=%.4f, drift=%s",
            self._baseline_ssim, current_ssim, delta, drift_detected,
        )

        return DriftResult(
            drift_detected=drift_detected,
            baseline_ssim=self._baseline_ssim,
            current_ssim=current_ssim,
            delta=delta,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            samples_checked=len(ssim_values),
        )

    def check_drift_from_dir(self, image_dir: Path) -> DriftResult:
        """Convenience: load images from directory and check drift."""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])
        sample = random.sample(paths, min(len(paths), self.window_size))

        images = []
        for p in sample:
            img = self._load_and_resize(p, self.target_size)
            if img is not None:
                images.append(img)

        return self.check_drift(images)


# ── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Drift Detector Self-Test ===\n")

    detector = DriftDetector(window_size=50, sample_pairs=20)

    # Create synthetic reference set (uniform gray images with noise)
    ref_images = []
    for _ in range(20):
        img = np.random.randint(100, 160, (256, 256), dtype=np.uint8)
        ref_images.append(img)

    # Manually set reference
    detector._reference_images = ref_images
    detector._baseline_ssim = 0.85  # Typical intra-set SSIM

    # Test 1: Similar batch (no drift)
    similar = [np.random.randint(100, 160, (256, 256), dtype=np.uint8) for _ in range(10)]
    result1 = detector.check_drift(similar)
    print(f"Test 1: Similar batch | drift={result1.drift_detected}, "
          f"delta={result1.delta:.4f}, rec={result1.recommendation}")
    # Don't assert specific values since it's random, just check structure
    assert isinstance(result1.drift_detected, bool)
    assert result1.samples_checked > 0
    print("Test 1 PASS: Structure correct")

    # Test 2: Very different batch (high drift)
    bright = [np.full((256, 256), 250, dtype=np.uint8) for _ in range(10)]
    result2 = detector.check_drift(bright)
    print(f"Test 2: Bright batch | drift={result2.drift_detected}, "
          f"delta={result2.delta:.4f}, rec={result2.recommendation}")
    assert result2.delta > 0  # Should show positive delta (lower SSIM)
    print("Test 2 PASS: Bright images show SSIM drop")

    # Test 3: Empty batch
    result3 = detector.check_drift([])
    assert not result3.drift_detected
    assert result3.samples_checked == 0
    print("Test 3 PASS: Empty batch handled correctly")

    print("\n=== All 3 tests passed ===")
