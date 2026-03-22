"""Merkezi YAML config yukleme."""

import yaml
from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict[str, Any]:
    """YAML dosyasini yukle, yoksa bos dict dondur."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_section(section: str, config_path: Path = None) -> dict[str, Any]:
    """Config'den belirli bir bolumu yukle."""
    if config_path is None:
        from src.common.constants import PROJECT_ROOT
        config_path = PROJECT_ROOT / "configs" / "phase2_config.yaml"
    return load_yaml(config_path).get(section, {})
