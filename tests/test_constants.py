"""Tests for src/common/constants.py."""

from src.common.constants import CLASS_NAMES, CLASS_NAMES_LIST, IMAGE_EXTS, PROJECT_ROOT


def test_class_names_has_three_entries():
    assert len(CLASS_NAMES) == 3


def test_class_names_values():
    assert CLASS_NAMES[0] == "screw"
    assert CLASS_NAMES[1] == "missing_screw"
    assert CLASS_NAMES[2] == "missing_component"


def test_class_names_list_matches_dict():
    assert CLASS_NAMES_LIST == list(CLASS_NAMES.values())


def test_image_exts_contains_common_formats():
    assert ".jpg" in IMAGE_EXTS
    assert ".png" in IMAGE_EXTS
    assert ".jpeg" in IMAGE_EXTS


def test_project_root_exists():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "src").exists()
