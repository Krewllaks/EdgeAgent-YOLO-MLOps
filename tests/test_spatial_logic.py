"""Tests for src/reasoning/spatial_logic.py — parametrized from _self_test scenarios."""

import pytest

from src.reasoning.spatial_logic import SpatialAnalyzer, Detection


@pytest.fixture
def analyzer():
    return SpatialAnalyzer(n_clusters=4)


IMG_SHAPE = (300, 400)


@pytest.mark.parametrize("name,detections,expected_verdict", [
    (
        "all_screws_ok",
        [
            Detection(0, 0.95, (50, 50, 100, 100)),
            Detection(0, 0.92, (50, 200, 100, 250)),
            Detection(0, 0.88, (300, 50, 350, 100)),
            Detection(0, 0.91, (300, 200, 350, 250)),
        ],
        "OK",
    ),
    (
        "left_missing_screw",
        [
            Detection(1, 0.85, (50, 50, 100, 100)),
            Detection(0, 0.90, (50, 200, 100, 250)),
            Detection(0, 0.88, (300, 50, 350, 100)),
            Detection(0, 0.91, (300, 200, 350, 250)),
        ],
        "missing_screw",
    ),
    (
        "left_both_missing_component",
        [
            Detection(1, 0.80, (50, 50, 100, 100)),
            Detection(1, 0.82, (50, 200, 100, 250)),
            Detection(0, 0.88, (300, 50, 350, 100)),
            Detection(0, 0.91, (300, 200, 350, 250)),
        ],
        "missing_component",
    ),
])
def test_spatial_verdicts(analyzer, name, detections, expected_verdict):
    result = analyzer.analyze_frame(detections, IMG_SHAPE)
    assert result.verdict == expected_verdict, f"Scenario '{name}': expected {expected_verdict}, got {result.verdict}"


def test_empty_detections(analyzer):
    result = analyzer.analyze_frame([], IMG_SHAPE)
    assert result.verdict is not None
