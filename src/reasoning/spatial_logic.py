"""Geometric Spatial Clustering Post-Processor.

Analyzes YOLO detection results using physical product geometry:
- Groups detections into spatial clusters (expected 4 screw positions)
- Assigns clusters to left/right sides
- Applies per-side decision matrix for final OK/NOK verdict

Per-Side Logic (2 screw positions per side):
    2 S  (both screws present)      -> side = S   (component present, OK)
    1 S + 1 MS (one screw missing)  -> side = MS  (component present, screw missing)
    2 MS (both screws missing)      -> side = MC  (component missing!)

Overall Decision Matrix:
    Left S  + Right S   -> OK
    Left MS + Right S   -> missing_screw (left side)
    Left S  + Right MS  -> missing_screw (right side)
    Left MS + Right MS  -> missing_screw (both sides)
    Left MC + Right any -> missing_component (left component gone)
    Right MC + Left any -> missing_component (right component gone)
    Any MC detection     -> missing_component (direct)

Usage (standalone test):
    python src/reasoning/spatial_logic.py --test
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = {0: "screw", 1: "missing_screw", 2: "missing_component"}
CLASS_IDS = {"screw": 0, "missing_screw": 1, "missing_component": 2}


@dataclass
class Detection:
    """Single YOLO detection."""
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel coords)

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, f"unknown_{self.class_id}")

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class SpatialCluster:
    """A group of spatially close detections."""
    cluster_id: int
    side: str = ""  # "left" or "right"
    center: Tuple[float, float] = (0.0, 0.0)
    detections: List[Detection] = field(default_factory=list)

    @property
    def dominant_class(self) -> str:
        if not self.detections:
            return "empty"
        # Count class occurrences; prioritize defect classes
        class_counts = {}
        for det in self.detections:
            name = det.class_name
            class_counts[name] = class_counts.get(name, 0) + 1

        # If any missing_component, that dominates
        if "missing_component" in class_counts:
            return "missing_component"
        # If any missing_screw, that dominates over screw
        if "missing_screw" in class_counts:
            return "missing_screw"
        return "screw"

    @property
    def avg_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)


@dataclass
class SpatialResult:
    """Final analysis result."""
    verdict: str  # "OK", "missing_screw", "missing_component"
    confidence: float
    reason: str
    clusters: List[SpatialCluster] = field(default_factory=list)
    left_status: str = ""
    right_status: str = ""
    detection_count: int = 0


class SpatialAnalyzer:
    """Geometric spatial clustering analyzer for screw detection.

    Supports dynamic rules via RuleEngine (configs/rules.yaml).
    If a product_name is given, rules are loaded from config instead of hard-coded.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        min_detections: int = 1,
        product_name: Optional[str] = None,
    ):
        """
        Args:
            n_clusters: Expected number of screw positions (default: 4).
                        Overridden by product rule if product_name is given.
            min_detections: Minimum detections to attempt clustering.
            product_name: If set, loads dynamic rules from configs/rules.yaml.
        """
        self._rule = None
        try:
            from src.reasoning.dynamic_rules import RuleEngine
            engine = RuleEngine()
            self._rule = engine.get_rule(product_name)
            self.n_clusters = self._rule.n_clusters
            self.min_detections = self._rule.clustering.get("min_detections", min_detections)
        except Exception:
            # Fallback: use provided values (backward compatibility)
            self.n_clusters = n_clusters
            self.min_detections = min_detections

    @property
    def max_total_detections(self) -> int:
        """Maximum allowed detections before rejecting as impossible."""
        if self._rule:
            return self._rule.max_total_detections
        return 8  # default: 4 screws * 2

    @property
    def max_missing_components(self) -> int:
        """Maximum missing component count."""
        if self._rule:
            return self._rule.max_missing_components
        return 2  # default

    def cluster_detections(
        self, detections: List[Detection], img_shape: Optional[Tuple[int, int]] = None
    ) -> List[SpatialCluster]:
        """Group detections into spatial clusters using simple K-means."""
        if len(detections) < self.min_detections:
            return []

        centers = np.array([d.center for d in detections])

        # Use simple K-means (avoid scipy dependency for basic clustering)
        k = min(self.n_clusters, len(detections))
        clusters = self._kmeans(centers, k, max_iter=50)

        result = []
        for i in range(k):
            mask = clusters == i
            cluster_dets = [d for d, m in zip(detections, mask) if m]
            if not cluster_dets:
                continue
            cluster_centers = centers[mask]
            center = tuple(cluster_centers.mean(axis=0))
            result.append(SpatialCluster(
                cluster_id=i,
                center=center,
                detections=cluster_dets,
            ))

        return result

    def _kmeans(self, points: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
        """Simple K-means clustering."""
        n = len(points)
        if n <= k:
            return np.arange(n)

        # Initialize centroids with K-means++ style
        rng = np.random.RandomState(42)
        centroids = [points[rng.randint(n)]]
        for _ in range(k - 1):
            dists = np.min([np.sum((points - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dists / dists.sum()
            idx = rng.choice(n, p=probs)
            centroids.append(points[idx])
        centroids = np.array(centroids)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # Assign
            dists = np.array([np.sum((points - c) ** 2, axis=1) for c in centroids])
            new_labels = dists.argmin(axis=0)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            # Update
            for i in range(k):
                mask = labels == i
                if mask.any():
                    centroids[i] = points[mask].mean(axis=0)

        return labels

    def assign_sides(self, clusters: List[SpatialCluster]) -> None:
        """Assign left/right side based on x-coordinate median."""
        if not clusters:
            return

        x_coords = [c.center[0] for c in clusters]
        median_x = np.median(x_coords)

        for cluster in clusters:
            cluster.side = "left" if cluster.center[0] <= median_x else "right"

        # If odd number, ensure at least one per side
        left_count = sum(1 for c in clusters if c.side == "left")
        if left_count == 0:
            clusters[0].side = "left"
        elif left_count == len(clusters):
            clusters[-1].side = "right"

    def apply_decision_matrix(
        self, clusters: List[SpatialCluster]
    ) -> Tuple[str, str, str, str]:
        """Apply the spatial decision matrix.

        Returns:
            (verdict, reason, left_status, right_status)
        """
        if not clusters:
            return ("missing_component", "Hicbir tespit yok - urun kontrolden gecemedi",
                    "unknown", "unknown")

        left_clusters = [c for c in clusters if c.side == "left"]
        right_clusters = [c for c in clusters if c.side == "right"]

        # Determine side statuses (includes per-side MC detection)
        left_status = self._side_status(left_clusters)
        right_status = self._side_status(right_clusters)

        # Decision matrix
        # MC on any side -> missing_component
        if left_status == "MC" or right_status == "MC":
            sides = []
            if left_status == "MC":
                sides.append("sol")
            if right_status == "MC":
                sides.append("sag")
            verdict = "missing_component"
            reason = (
                f"Tum vidalar eksik ({', '.join(sides)} taraf) - "
                f"missing_component kesin (Sol: {left_status}, Sag: {right_status})"
            )
        elif left_status == "S" and right_status == "S":
            verdict = "OK"
            reason = "Tum vidalar mevcut (Sol: OK, Sag: OK)"
        elif left_status == "MS" and right_status == "MS":
            verdict = "missing_screw"
            reason = f"Her iki tarafta vida eksik (Sol: {left_status}, Sag: {right_status})"
        elif left_status == "MS":
            verdict = "missing_screw"
            reason = f"Sol tarafta vida eksik (Sol: {left_status}, Sag: {right_status})"
        elif right_status == "MS":
            verdict = "missing_screw"
            reason = f"Sag tarafta vida eksik (Sol: {left_status}, Sag: {right_status})"
        else:
            verdict = "OK"
            reason = f"Belirsiz durum, varsayilan OK (Sol: {left_status}, Sag: {right_status})"

        return (verdict, reason, left_status, right_status)

    def _side_status(self, side_clusters: List[SpatialCluster]) -> str:
        """Determine the status of one side (left or right).

        Per-side logic (expects 2 screw positions per side):
            2 S  -> "S"  (all screws present, component OK)
            1 S + 1 MS -> "MS" (component present but screw missing)
            2 MS -> "MC" (no screws at all -> component missing)

        Returns: "S", "MS", "MC", or "unknown"
        """
        if not side_clusters:
            return "unknown"

        all_dets = [d for c in side_clusters for d in c.detections]
        if not all_dets:
            return "unknown"

        ms_count = sum(1 for d in all_dets if d.class_name == "missing_screw")
        s_count = sum(1 for d in all_dets if d.class_name == "screw")
        mc_count = sum(1 for d in all_dets if d.class_name == "missing_component")

        # Direct missing_component detection on this side
        if mc_count > 0:
            return "MC"

        # If at least one screw is present, component is there
        if s_count > 0 and ms_count == 0:
            return "S"
        if s_count > 0 and ms_count > 0:
            return "MS"
        # All detections are missing_screw, no screw at all -> component missing
        if ms_count > 0 and s_count == 0:
            return "MC"

        return "unknown"

    def analyze_frame(
        self,
        detections: List[Detection],
        img_shape: Optional[Tuple[int, int]] = None,
    ) -> SpatialResult:
        """Full analysis pipeline: domain check -> cluster -> assign sides -> decision matrix.

        Args:
            detections: List of YOLO detections
            img_shape: (height, width) of the image

        Returns:
            SpatialResult with verdict, clusters, and reasoning
        """
        if not detections:
            return SpatialResult(
                verdict="missing_component",
                confidence=0.0,
                reason="Hicbir nesne tespit edilemedi",
                detection_count=0,
            )

        # Domain constraint check: reject impossible outputs
        if len(detections) > self.max_total_detections:
            rule_name = self._rule.name if self._rule else "default"
            return SpatialResult(
                verdict="rejected",
                confidence=0.0,
                reason=(
                    f"Tespit sayisi ({len(detections)}) domain kisitiyla uyumsuz "
                    f"(max={self.max_total_detections}, rule='{rule_name}'). "
                    f"Auto-tune onerisi: Canny parametrelerini yeniden ayarlayin."
                ),
                detection_count=len(detections),
            )

        clusters = self.cluster_detections(detections, img_shape)
        self.assign_sides(clusters)
        verdict, reason, left_status, right_status = self.apply_decision_matrix(clusters)

        avg_conf = np.mean([d.confidence for d in detections]) if detections else 0.0

        return SpatialResult(
            verdict=verdict,
            confidence=float(avg_conf),
            reason=reason,
            clusters=clusters,
            left_status=left_status,
            right_status=right_status,
            detection_count=len(detections),
        )


def detections_from_yolo_result(result) -> List[Detection]:
    """Convert ultralytics YOLO result to Detection list."""
    detections = []
    boxes = result.boxes
    for i in range(len(boxes)):
        det = Detection(
            class_id=int(boxes.cls[i]),
            confidence=float(boxes.conf[i]),
            bbox=tuple(boxes.xyxy[i].tolist()),
        )
        detections.append(det)
    return detections


def _run_self_test():
    """Self-test with synthetic data."""
    analyzer = SpatialAnalyzer(n_clusters=4)

    # Test 1: All screws present (2S + 2S) -> OK
    dets_ok = [
        Detection(0, 0.95, (50, 50, 100, 100)),    # left-top screw
        Detection(0, 0.92, (50, 200, 100, 250)),    # left-bottom screw
        Detection(0, 0.88, (300, 50, 350, 100)),    # right-top screw
        Detection(0, 0.91, (300, 200, 350, 250)),   # right-bottom screw
    ]
    result = analyzer.analyze_frame(dets_ok, (300, 400))
    print(f"Test 1 (2S+2S): verdict={result.verdict}, reason={result.reason}")
    assert result.verdict == "OK", f"Expected OK, got {result.verdict}"

    # Test 2: Left 1S+1MS, Right 2S -> missing_screw
    dets_ms = [
        Detection(1, 0.85, (50, 50, 100, 100)),     # left-top missing_screw
        Detection(0, 0.90, (50, 200, 100, 250)),     # left-bottom screw
        Detection(0, 0.88, (300, 50, 350, 100)),     # right-top screw
        Detection(0, 0.91, (300, 200, 350, 250)),    # right-bottom screw
    ]
    result = analyzer.analyze_frame(dets_ms, (300, 400))
    print(f"Test 2 (1S+1MS | 2S): verdict={result.verdict}, reason={result.reason}")
    assert result.verdict == "missing_screw", f"Expected missing_screw, got {result.verdict}"

    # Test 3: Left 2MS -> missing_component (even if right is 2S)
    dets_left_mc = [
        Detection(1, 0.80, (50, 50, 100, 100)),     # left-top missing
        Detection(1, 0.82, (50, 200, 100, 250)),     # left-bottom missing
        Detection(0, 0.88, (300, 50, 350, 100)),     # right-top screw
        Detection(0, 0.91, (300, 200, 350, 250)),    # right-bottom screw
    ]
    result = analyzer.analyze_frame(dets_left_mc, (300, 400))
    print(f"Test 3 (2MS | 2S): verdict={result.verdict}, reason={result.reason}")
    assert result.verdict == "missing_component", f"Expected missing_component, got {result.verdict}"

    # Test 4: Both sides 2MS -> missing_component
    dets_both_mc = [
        Detection(1, 0.80, (50, 50, 100, 100)),     # left missing
        Detection(1, 0.82, (50, 200, 100, 250)),     # left missing
        Detection(1, 0.78, (300, 50, 350, 100)),     # right missing
        Detection(1, 0.75, (300, 200, 350, 250)),    # right missing
    ]
    result = analyzer.analyze_frame(dets_both_mc, (300, 400))
    print(f"Test 4 (2MS+2MS): verdict={result.verdict}, reason={result.reason}")
    assert result.verdict == "missing_component", f"Expected missing_component, got {result.verdict}"

    # Test 5: Both sides 1S+1MS -> missing_screw (component still present)
    dets_both_ms = [
        Detection(1, 0.80, (50, 50, 100, 100)),     # left-top missing
        Detection(0, 0.82, (50, 200, 100, 250)),     # left-bottom screw
        Detection(1, 0.78, (300, 50, 350, 100)),     # right-top missing
        Detection(0, 0.75, (300, 200, 350, 250)),    # right-bottom screw
    ]
    result = analyzer.analyze_frame(dets_both_ms, (300, 400))
    print(f"Test 5 (1S+1MS | 1S+1MS): verdict={result.verdict}, reason={result.reason}")
    assert result.verdict == "missing_screw", f"Expected missing_screw, got {result.verdict}"

    # Test 6: No detections
    result = analyzer.analyze_frame([], (300, 400))
    print(f"Test 6 (empty): verdict={result.verdict}, reason={result.reason}")
    assert result.verdict == "missing_component"

    # Test 7: Direct MC detection
    dets_direct_mc = [
        Detection(2, 0.90, (100, 100, 300, 300)),   # missing_component
        Detection(0, 0.85, (50, 50, 100, 100)),      # screw
    ]
    result = analyzer.analyze_frame(dets_direct_mc, (400, 400))
    print(f"Test 7 (direct MC): verdict={result.verdict}, reason={result.reason}")
    assert result.verdict == "missing_component"

    print("\n[OK] All spatial logic tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run self-test")
    args = parser.parse_args()

    if args.test:
        _run_self_test()
    else:
        print("Usage: python src/reasoning/spatial_logic.py --test")
