"""Proje geneli sabitler — tek kaynak (Single Source of Truth)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES: dict[int, str] = {0: "screw", 1: "missing_screw", 2: "missing_component"}
CLASS_NAMES_LIST: list[str] = ["screw", "missing_screw", "missing_component"]
IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
