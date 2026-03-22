"""Tests for src/common/config.py."""

from pathlib import Path

from src.common.config import load_yaml, load_section


def test_load_yaml_nonexistent_returns_empty():
    result = load_yaml(Path("nonexistent_file_xyz.yaml"))
    assert result == {}


def test_load_yaml_reads_production_config():
    from src.common.constants import PROJECT_ROOT
    cfg = load_yaml(PROJECT_ROOT / "configs" / "production_config.yaml")
    assert "camera" in cfg
    assert "model" in cfg


def test_load_section_returns_dict():
    result = load_section("nonexistent_section_xyz")
    assert isinstance(result, dict)
    assert result == {}
