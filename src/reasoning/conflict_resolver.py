"""Conflict Resolver - YOLO vs Spatial vs VLM arbitration.

When YOLO detection results and Spatial Logic disagree, this module
uses the "VLM as Judge" principle to produce a final verdict.

Decision flow:
  1. YOLO + Spatial agree     -> consensus verdict (VLM skipped for speed)
  2. YOLO + Spatial disagree  -> VLM is final arbiter
  3. VLM unavailable + conflict -> Spatial wins (conservative, lower FP)

Usage:
    from src.reasoning.conflict_resolver import ConflictResolver
    resolver = ConflictResolver()
    verdict = resolver.resolve(yolo_dets, spatial_result, vlm_result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.common.config import load_section
from src.reasoning.rca_templates import get_rca

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "phase2_config.yaml"


@dataclass
class FinalVerdict:
    """Final quality control verdict after all reasoning stages."""

    verdict: str  # "ok", "missing_screw", "missing_component"
    source: str  # "consensus", "vlm", "spatial", "yolo"
    confidence: float
    conflict_detected: bool
    reasoning: str
    rca_text: str


class ConflictResolver:
    """Resolves conflicts between YOLO, Spatial Logic, and VLM."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG):
        self._config = load_section("conflict_resolver", config_path)

    @staticmethod
    def _extract_yolo_verdict(detections: list[dict]) -> tuple[str, float]:
        """Derive verdict from raw YOLO detections.

        Args:
            detections: List of dicts with 'class_name' and 'confidence' keys.

        Returns:
            (verdict, avg_confidence)
        """
        if not detections:
            return "no_detection", 0.0

        class_names = [d.get("class_name", d.get("class", "")) for d in detections]
        confs = [d.get("confidence", d.get("conf", 0.0)) for d in detections]
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        if "missing_component" in class_names:
            return "missing_component", avg_conf
        if "missing_screw" in class_names:
            return "missing_screw", avg_conf
        return "ok", avg_conf

    def resolve(
        self,
        yolo_detections: list[dict],
        spatial_verdict: Optional[str],
        spatial_side: Optional[str] = None,
        vlm_result=None,
    ) -> FinalVerdict:
        """Produce final verdict from all reasoning sources.

        Args:
            yolo_detections: Raw YOLO detection dicts.
            spatial_verdict: Spatial logic verdict ("ok", "missing_screw",
                           "missing_component") or None if not run.
            spatial_side: Which side had the issue ("left", "right", "both").
            vlm_result: ReasoningResult from VLMReasoner, or None.

        Returns:
            FinalVerdict with consensus or arbitrated decision.
        """
        yolo_verdict, yolo_conf = self._extract_yolo_verdict(yolo_detections)

        # If spatial logic wasn't run, use YOLO directly
        if spatial_verdict is None:
            rca = get_rca(yolo_verdict, side=spatial_side)
            return FinalVerdict(
                verdict=yolo_verdict,
                source="yolo",
                confidence=yolo_conf,
                conflict_detected=False,
                reasoning=f"Spatial analiz yapilmadi, YOLO sonucu kullanildi: {yolo_verdict}",
                rca_text=rca,
            )

        # Check consensus
        consensus = yolo_verdict == spatial_verdict

        if consensus:
            # YOLO and Spatial agree - no need for VLM
            conf = max(yolo_conf, 0.7)  # Consensus boosts confidence
            rca = get_rca(spatial_verdict, side=spatial_side)
            return FinalVerdict(
                verdict=spatial_verdict,
                source="consensus",
                confidence=conf,
                conflict_detected=False,
                reasoning=(
                    f"YOLO ve Mekansal Analiz ayni sonuca ulasti: {spatial_verdict}"
                ),
                rca_text=rca,
            )

        # Conflict detected
        logger.info(
            "Conflict: YOLO=%s vs Spatial=%s", yolo_verdict, spatial_verdict
        )

        if vlm_result is not None and vlm_result.defect_type is not None:
            # VLM as Judge
            vlm_override = vlm_result.defect_type != spatial_verdict
            rca = get_rca(
                vlm_result.defect_type,
                side=spatial_side,
                vlm_override=vlm_override,
            )
            return FinalVerdict(
                verdict=vlm_result.defect_type,
                source="vlm",
                confidence=vlm_result.confidence_estimate,
                conflict_detected=True,
                reasoning=(
                    f"YOLO ({yolo_verdict}) ve Mekansal ({spatial_verdict}) celisti. "
                    f"VLM karari: {vlm_result.defect_type}. "
                    f"VLM aciklamasi: {vlm_result.reasoning}"
                ),
                rca_text=rca,
            )

        # VLM unavailable or unparseable -> Spatial wins (conservative)
        rca = get_rca(spatial_verdict, side=spatial_side)
        return FinalVerdict(
            verdict=spatial_verdict,
            source="spatial",
            confidence=0.5,  # Lower confidence due to unresolved conflict
            conflict_detected=True,
            reasoning=(
                f"YOLO ({yolo_verdict}) ve Mekansal ({spatial_verdict}) celisti. "
                f"VLM mevcut degil/basarisiz, konservatif karar: {spatial_verdict}"
            ),
            rca_text=rca,
        )


# ── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dataclasses import dataclass as _dc

    @_dc
    class _MockVLM:
        defect_type: str | None
        confidence_estimate: float
        reasoning: str

    print("=== Conflict Resolver Self-Test ===\n")
    resolver = ConflictResolver()

    # Test 1: Consensus (YOLO=ok, Spatial=ok)
    v = resolver.resolve(
        [{"class_name": "screw", "confidence": 0.95}],
        spatial_verdict="ok",
    )
    assert v.verdict == "ok" and v.source == "consensus" and not v.conflict_detected
    print("Test 1 PASS: Consensus ok")

    # Test 2: Consensus missing_screw
    v = resolver.resolve(
        [{"class_name": "missing_screw", "confidence": 0.8}],
        spatial_verdict="missing_screw",
        spatial_side="left",
    )
    assert v.verdict == "missing_screw" and v.source == "consensus"
    assert "Sol" in v.rca_text
    print("Test 2 PASS: Consensus missing_screw with left RCA")

    # Test 3: Conflict + VLM resolves
    vlm = _MockVLM("missing_component", 0.85, "Bracket is completely absent")
    v = resolver.resolve(
        [{"class_name": "screw", "confidence": 0.9}],
        spatial_verdict="missing_component",
        spatial_side="right",
        vlm_result=vlm,
    )
    assert v.verdict == "missing_component" and v.source == "vlm"
    assert v.conflict_detected
    print("Test 3 PASS: Conflict resolved by VLM")

    # Test 4: Conflict + VLM unavailable -> Spatial wins
    v = resolver.resolve(
        [{"class_name": "screw", "confidence": 0.9}],
        spatial_verdict="missing_screw",
        spatial_side="both",
        vlm_result=None,
    )
    assert v.verdict == "missing_screw" and v.source == "spatial"
    assert v.conflict_detected and v.confidence == 0.5
    print("Test 4 PASS: Conflict without VLM -> Spatial wins (conservative)")

    # Test 5: No spatial -> YOLO only
    v = resolver.resolve(
        [{"class_name": "missing_component", "confidence": 0.7}],
        spatial_verdict=None,
    )
    assert v.verdict == "missing_component" and v.source == "yolo"
    print("Test 5 PASS: No spatial -> YOLO verdict")

    # Test 6: Empty detections
    v = resolver.resolve([], spatial_verdict="missing_component")
    assert v.conflict_detected  # no_detection vs missing_component
    print("Test 6 PASS: Empty detections -> conflict detected")

    # Test 7: VLM with unparseable result (None defect_type) -> Spatial wins
    vlm_bad = _MockVLM(None, 0.0, "Cannot determine")
    v = resolver.resolve(
        [{"class_name": "screw", "confidence": 0.9}],
        spatial_verdict="missing_screw",
        vlm_result=vlm_bad,
    )
    assert v.verdict == "missing_screw" and v.source == "spatial"
    print("Test 7 PASS: Unparseable VLM -> Spatial fallback")

    print("\n=== All 7 tests passed ===")
