"""Tests for src/reasoning/rca_templates.py."""

from src.reasoning.rca_templates import get_rca, RCA_TEMPLATES


def test_rca_templates_has_10_keys():
    assert len(RCA_TEMPLATES) >= 10


def test_all_templates_non_empty():
    for key, text in RCA_TEMPLATES.items():
        assert text.strip(), f"RCA template '{key}' is empty"


def test_get_rca_known_key():
    result = get_rca("ok_all_present")
    assert result
    assert isinstance(result, str)


def test_get_rca_unknown_key_returns_fallback():
    result = get_rca("nonexistent_key_xyz")
    assert isinstance(result, str)
