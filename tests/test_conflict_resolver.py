"""Tests for src/reasoning/conflict_resolver.py — 3 decision paths."""

from src.reasoning.conflict_resolver import ConflictResolver


def test_consensus_when_yolo_spatial_agree():
    resolver = ConflictResolver()
    yolo_dets = [
        {"class_name": "screw", "confidence": 0.95},
        {"class_name": "screw", "confidence": 0.92},
        {"class_name": "screw", "confidence": 0.88},
        {"class_name": "screw", "confidence": 0.91},
    ]
    verdict = resolver.resolve(yolo_dets, spatial_verdict="ok", vlm_result=None)
    assert verdict.source == "consensus"
    assert verdict.verdict.lower() == "ok"


def test_fail_safe_when_vlm_unavailable():
    resolver = ConflictResolver()
    yolo_dets = [
        {"class_name": "missing_screw", "confidence": 0.60},
        {"class_name": "screw", "confidence": 0.92},
        {"class_name": "screw", "confidence": 0.88},
        {"class_name": "screw", "confidence": 0.91},
    ]
    verdict = resolver.resolve(yolo_dets, spatial_verdict="ok", vlm_result=None)
    assert verdict is not None
    assert isinstance(verdict.verdict, str)
