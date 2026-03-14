"""Root Cause Analysis (RCA) Templates - Turkish industrial defect explanations.

Provides 10 pre-defined RCA templates for quick defect reporting.
Each template maps to a specific combination of spatial analysis result
and VLM reasoning output.

Usage:
    from src.reasoning.rca_templates import get_rca
    text = get_rca("missing_screw", side="left")
"""

from __future__ import annotations

from typing import Optional


# ── Template Definitions ─────────────────────────────────────────────

RCA_TEMPLATES: dict[str, str] = {
    "missing_screw_left": (
        "Sol tarafta vida eksikligi tespit edildi. "
        "Montaj hattinda vida besleme unitesinin kontrolu oneriliyor. "
        "Olasi neden: Vida besleyici bosluk, robot kol pozisyon sapmasi."
    ),
    "missing_screw_right": (
        "Sag tarafta vida eksikligi tespit edildi. "
        "Montaj hattinda vida besleme unitesinin kontrolu oneriliyor. "
        "Olasi neden: Vida besleyici bosluk, robot kol pozisyon sapmasi."
    ),
    "missing_screw_both": (
        "Her iki tarafta vida eksikligi tespit edildi. "
        "Montaj hatti tamamen durdurulmali ve vida besleme sistemi kontrol edilmeli. "
        "Olasi neden: Vida deposu bos, besleyici mekanizma arizasi."
    ),
    "missing_component_left": (
        "Sol taraftaki komponent (parca) tamamen eksik. "
        "Bu kritik bir hatadir - uretim hatti durdurulmali. "
        "Olasi neden: Parca besleme sistemi arizasi, onceki istasyonda montaj hatasi."
    ),
    "missing_component_right": (
        "Sag taraftaki komponent (parca) tamamen eksik. "
        "Bu kritik bir hatadir - uretim hatti durdurulmali. "
        "Olasi neden: Parca besleme sistemi arizasi, onceki istasyonda montaj hatasi."
    ),
    "missing_component_both": (
        "Her iki taraftaki komponentler eksik. "
        "ACIL: Uretim hatti derhal durdurulmali. "
        "Olasi neden: Parca besleme sistemi tamamen devre disi, malzeme bitmis olabilir."
    ),
    "low_confidence_screw": (
        "Vida tespiti dusuk guvenle yapildi (conf < 0.40). "
        "Gorsel kalite sorunu olabilir (yansima, bulaniklik). "
        "Onerilen aksiyon: Manuel kontrol, kamera aydınlatma ayari."
    ),
    "vlm_override_spatial": (
        "VLM analizi, mekansal kume analizinden farkli bir sonuc uretti. "
        "VLM karari nihai olarak kabul edildi (VLM-as-Judge prensibi). "
        "Operatorun gorsel kontrolu onerilir."
    ),
    "no_detection": (
        "Goruntude hicbir nesne tespit edilemedi. "
        "Olasi nedenler: Parca yok, kamera gorus alani bos, "
        "agir yansima veya karartma. Kamera pozisyonunu kontrol edin."
    ),
    "ok_all_present": (
        "Tum vidalar ve komponentler dogru pozisyonda tespit edildi. "
        "Parca kalite kontrolden gecti - uretim hattina devam edilebilir."
    ),
}


def get_rca(
    verdict: str,
    side: Optional[str] = None,
    vlm_override: bool = False,
    low_confidence: bool = False,
) -> str:
    """Get RCA template text for a given verdict.

    Args:
        verdict: "ok", "missing_screw", "missing_component", "no_detection"
        side: "left", "right", "both", or None
        vlm_override: True if VLM overrode spatial logic
        low_confidence: True if detections were below confidence threshold

    Returns:
        Turkish RCA explanation string
    """
    # Special cases first
    if vlm_override:
        return RCA_TEMPLATES["vlm_override_spatial"]

    if low_confidence and verdict == "missing_screw":
        return RCA_TEMPLATES["low_confidence_screw"]

    if verdict == "no_detection":
        return RCA_TEMPLATES["no_detection"]

    if verdict == "ok":
        return RCA_TEMPLATES["ok_all_present"]

    # Side-specific templates
    if side and verdict in ("missing_screw", "missing_component"):
        key = f"{verdict}_{side}"
        if key in RCA_TEMPLATES:
            return RCA_TEMPLATES[key]

    # Fallback: try without side
    for s in ("both", "left", "right"):
        key = f"{verdict}_{s}"
        if key in RCA_TEMPLATES:
            return RCA_TEMPLATES[key]

    return f"Bilinmeyen durum: {verdict} (side={side})"


# ── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== RCA Templates Self-Test ===\n")

    assert "Sol" in get_rca("missing_screw", side="left")
    print("Test 1 PASS: missing_screw left")

    assert "Sag" in get_rca("missing_screw", side="right")
    print("Test 2 PASS: missing_screw right")

    assert "Her iki" in get_rca("missing_screw", side="both")
    print("Test 3 PASS: missing_screw both")

    assert "ACIL" in get_rca("missing_component", side="both")
    print("Test 4 PASS: missing_component both (ACIL)")

    assert "VLM" in get_rca("missing_screw", vlm_override=True)
    print("Test 5 PASS: vlm_override")

    assert "dusuk guven" in get_rca("missing_screw", low_confidence=True)
    print("Test 6 PASS: low_confidence")

    assert "Tum vidalar" in get_rca("ok")
    print("Test 7 PASS: ok")

    assert "tespit edilemedi" in get_rca("no_detection")
    print("Test 8 PASS: no_detection")

    # Unknown verdict fallback
    result = get_rca("unknown_thing")
    assert "Bilinmeyen" in result
    print("Test 9 PASS: unknown fallback")

    print("\n=== All 9 tests passed ===")
