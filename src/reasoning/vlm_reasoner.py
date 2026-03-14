"""PaliGemma VLM Reasoning Engine - Warm Standby Pattern.

Loads PaliGemma-3B in NF4 quantization and keeps it warm in VRAM.
Accepts cropped image regions from YOLO detections and returns
structured defect analysis.

VRAM budget (RTX 3050 4GB):
  YOLO eval:     ~500 MB
  PaliGemma NF4: ~2800 MB
  Overhead:      ~400 MB
  Total:         ~3700 MB (fits in 4GB)

Usage:
    from src.reasoning.vlm_reasoner import VLMReasoner

    reasoner = VLMReasoner()
    reasoner.load_model()
    result = reasoner.reason(cropped_image)
"""

from __future__ import annotations

import gc
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "phase2_config.yaml"

# Valid defect types that the VLM can report
VALID_DEFECT_TYPES = {"missing_screw", "missing_component", "ok"}

# Fallback prompt if config is missing
_DEFAULT_PROMPT = (
    "Analyze this industrial part image. "
    "Is there a missing screw, missing component, or is everything OK? "
    "Answer in this exact format:\n"
    "DEFECT_TYPE: [missing_screw|missing_component|ok]\n"
    "REASON: [your explanation]"
)


@dataclass
class ReasoningResult:
    """Structured output from VLM inference."""

    raw_text: str
    defect_type: Optional[str]  # missing_screw | missing_component | ok | None
    confidence_estimate: float  # 0.0 if unparseable
    reasoning: str  # Human-readable explanation
    latency_ms: float
    model_id: str


def _parse_vlm_output(text: str) -> tuple[Optional[str], float, str]:
    """Parse structured VLM output into defect_type, confidence, reasoning.

    Expected format:
        DEFECT_TYPE: missing_screw
        REASON: The screw hole is empty...

    Returns (defect_type, confidence, reasoning).
    If parsing fails, returns (None, 0.0, raw_text).
    """
    defect_type = None
    reasoning = text.strip()
    confidence = 0.0

    # Try to extract DEFECT_TYPE
    dt_match = re.search(
        r"DEFECT_TYPE\s*:\s*(missing_screw|missing_component|ok)",
        text,
        re.IGNORECASE,
    )
    if dt_match:
        defect_type = dt_match.group(1).lower()
        if defect_type in VALID_DEFECT_TYPES:
            confidence = 0.85  # Structured response -> reasonable confidence
        else:
            defect_type = None

    # Try to extract REASON
    reason_match = re.search(r"REASON\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reasoning = reason_match.group(1).strip()

    # If no structured format found, try keyword detection as fallback
    if defect_type is None:
        text_lower = text.lower()
        # Check component-level missing first (more severe)
        component_kw = [
            "missing_component", "component missing", "component absent",
            "bracket missing", "bracket absent", "part missing", "part absent",
        ]
        screw_kw = [
            "missing_screw", "screw missing", "screw absent",
            "screw is missing", "missing screw", "empty screw hole",
            "screw hole is empty", "no screw",
        ]
        if any(kw in text_lower for kw in component_kw):
            defect_type = "missing_component"
            confidence = 0.5  # Lower confidence for unstructured
        elif any(kw in text_lower for kw in screw_kw):
            defect_type = "missing_screw"
            confidence = 0.5
        elif "ok" in text_lower and "missing" not in text_lower:
            defect_type = "ok"
            confidence = 0.5

    return defect_type, confidence, reasoning


class VLMReasoner:
    """PaliGemma 3B NF4 inference engine with Warm Standby pattern.

    The model stays loaded in VRAM after first load. This avoids the
    8-15 second cold-start penalty that would violate the 2s latency budget.
    """

    def __init__(self, config_path: Path = DEFAULT_CONFIG):
        self._config = self._load_config(config_path)
        self._model = None
        self._processor = None
        self._loaded = False
        self._lock = threading.Lock()
        self._device = None

    @staticmethod
    def _load_config(path: Path) -> dict:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("vlm", {})
        logger.warning("Phase 2 config not found at %s, using defaults", path)
        return {}

    @property
    def model_id(self) -> str:
        return self._config.get("model_id", "google/paligemma-3b-mix-224")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> None:
        """Load PaliGemma 3B with NF4 quantization into VRAM.

        After loading, runs warmup inferences to fill CUDA caches and
        avoid first-inference latency spikes.
        """
        if self._loaded:
            logger.info("VLM already loaded, skipping")
            return

        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        quant_type = self._config.get("quantization", "nf4")
        model_id = self.model_id

        logger.info("Loading VLM: %s (quantization=%s)", model_id, quant_type)

        # Determine device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if quant_type == "nf4" and self._device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self._model = PaliGemmaForConditionalGeneration.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            except ImportError:
                logger.warning(
                    "bitsandbytes not available, loading in float16 (higher VRAM)"
                )
                self._model = PaliGemmaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        else:
            # CPU or non-NF4: load in float32
            self._model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
            )

        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(model_id)

        # Warmup inferences to fill CUDA caches
        warmup_runs = self._config.get("warmup_runs", 2)
        self._warmup(warmup_runs)

        self._loaded = True

        # Log VRAM usage
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            logger.info("VLM loaded. VRAM usage: %.1f MB", vram_mb)

    def _warmup(self, n: int = 2) -> None:
        """Run dummy inferences to warm up CUDA caches."""
        from PIL import Image

        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        for i in range(n):
            try:
                inputs = self._processor(
                    text="Describe this image.",
                    images=dummy,
                    return_tensors="pt",
                )
                # Move inputs to model device
                inputs = {
                    k: v.to(self._model.device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }
                with __import__("torch").no_grad():
                    self._model.generate(**inputs, max_new_tokens=10)
                logger.debug("Warmup %d/%d complete", i + 1, n)
            except Exception as e:
                logger.warning("Warmup %d failed: %s", i + 1, e)

    def unload_model(self) -> None:
        """Unload model from VRAM and free memory."""
        import torch

        self._model = None
        self._processor = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VLM unloaded, VRAM freed")

    def ensure_loaded(self) -> None:
        """Thread-safe check: load model if not already loaded."""
        if self._loaded:
            return
        with self._lock:
            if not self._loaded:
                self.load_model()

    @staticmethod
    def crop_region(
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
        padding_ratio: float = 0.20,
    ) -> np.ndarray:
        """Crop image region around bbox with padding.

        Args:
            image: HWC numpy array (RGB or BGR)
            bbox: (x1, y1, x2, y2) pixel coordinates
            padding_ratio: Expand bbox by this fraction on each side

        Returns:
            Cropped numpy array
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1

        # Add padding
        pad_x = bw * padding_ratio
        pad_y = bh * padding_ratio
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))

        return image[y1:y2, x1:x2].copy()

    def reason(
        self,
        image: np.ndarray,
        prompt: str | None = None,
    ) -> ReasoningResult:
        """Run VLM inference on a cropped image region.

        Args:
            image: HWC numpy array (RGB). Typically a cropped defect region.
            prompt: Override prompt. If None, uses config template.

        Returns:
            ReasoningResult with parsed defect analysis.
        """
        import torch
        from PIL import Image as PILImage

        self.ensure_loaded()

        if prompt is None:
            prompt = self._config.get("prompt_template", _DEFAULT_PROMPT).strip()

        # Convert numpy to PIL
        if image.ndim == 3 and image.shape[2] == 3:
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = PILImage.fromarray(image).convert("RGB")

        # Inference
        t0 = time.perf_counter()
        try:
            inputs = self._processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt",
            )
            inputs = {
                k: v.to(self._model.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            max_tokens = self._config.get("max_new_tokens", 256)
            do_sample = self._config.get("do_sample", False)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                )

            # Decode only the generated tokens (skip input tokens)
            input_len = inputs["input_ids"].shape[1]
            raw_text = self._processor.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during VLM inference - returning empty result")
            latency = (time.perf_counter() - t0) * 1000
            return ReasoningResult(
                raw_text="[OOM] CUDA out of memory",
                defect_type=None,
                confidence_estimate=0.0,
                reasoning="VLM inference failed due to insufficient VRAM",
                latency_ms=latency,
                model_id=self.model_id,
            )
        except Exception as e:
            logger.error("VLM inference error: %s", e)
            latency = (time.perf_counter() - t0) * 1000
            return ReasoningResult(
                raw_text=f"[ERROR] {e}",
                defect_type=None,
                confidence_estimate=0.0,
                reasoning=f"VLM inference failed: {e}",
                latency_ms=latency,
                model_id=self.model_id,
            )

        latency = (time.perf_counter() - t0) * 1000

        # Parse structured output
        defect_type, confidence, reasoning = _parse_vlm_output(raw_text)

        # Warn if latency exceeds budget
        max_latency = self._config.get("max_latency_ms", 2000)
        if latency > max_latency:
            logger.warning(
                "VLM latency %.0fms exceeds budget %dms", latency, max_latency
            )

        return ReasoningResult(
            raw_text=raw_text,
            defect_type=defect_type,
            confidence_estimate=confidence,
            reasoning=reasoning,
            latency_ms=latency,
            model_id=self.model_id,
        )

    def get_vram_usage_mb(self) -> float:
        """Return current VRAM usage in MB."""
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0


# ── Self-test ────────────────────────────────────────────────────────

def _self_test() -> None:
    """Test VLM output parsing logic (no model required)."""
    print("=== VLM Reasoner Self-Test ===\n")

    # Test 1: Well-formed output
    text1 = "DEFECT_TYPE: missing_screw\nREASON: The screw hole on the left is empty."
    dt, conf, reason = _parse_vlm_output(text1)
    assert dt == "missing_screw", f"Test 1 fail: {dt}"
    assert conf == 0.85, f"Test 1 conf fail: {conf}"
    assert "screw hole" in reason, f"Test 1 reason fail: {reason}"
    print("Test 1 PASS: Structured missing_screw parsed correctly")

    # Test 2: Well-formed OK
    text2 = "DEFECT_TYPE: ok\nREASON: All screws and components are present."
    dt, conf, reason = _parse_vlm_output(text2)
    assert dt == "ok", f"Test 2 fail: {dt}"
    assert conf == 0.85
    print("Test 2 PASS: Structured ok parsed correctly")

    # Test 3: Unstructured fallback
    text3 = "I can see that a screw is missing from the bottom left corner."
    dt, conf, reason = _parse_vlm_output(text3)
    assert dt == "missing_screw", f"Test 3 fail: {dt}"
    assert conf == 0.5  # Lower confidence for unstructured
    print("Test 3 PASS: Unstructured missing_screw detected with lower confidence")

    # Test 4: Completely unparseable
    text4 = "The image shows a metal surface with some texture."
    dt, conf, reason = _parse_vlm_output(text4)
    assert dt is None, f"Test 4 fail: {dt}"
    assert conf == 0.0
    print("Test 4 PASS: Unparseable output returns None with confidence 0.0")

    # Test 5: missing_component structured
    text5 = "DEFECT_TYPE: missing_component\nREASON: The entire bracket is absent."
    dt, conf, reason = _parse_vlm_output(text5)
    assert dt == "missing_component", f"Test 5 fail: {dt}"
    assert conf == 0.85
    print("Test 5 PASS: missing_component parsed correctly")

    # Test 6: Crop region
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    crop = VLMReasoner.crop_region(img, (100, 100, 200, 200), padding_ratio=0.20)
    # bbox=100x100, 20% pad = 20px each side -> 140x140
    assert crop.shape[0] == 140, f"Test 6 h fail: {crop.shape}"
    assert crop.shape[1] == 140, f"Test 6 w fail: {crop.shape}"
    print("Test 6 PASS: crop_region with 20% padding correct")

    # Test 7: Crop region at edge (clamp to 0)
    crop2 = VLMReasoner.crop_region(img, (0, 0, 50, 50), padding_ratio=0.20)
    # bbox=50x50, 20% pad = 10px. Top/left clamped to 0 -> 0:60, 0:60
    assert crop2.shape[0] == 60, f"Test 7 h fail: {crop2.shape}"
    assert crop2.shape[1] == 60, f"Test 7 w fail: {crop2.shape}"
    print("Test 7 PASS: crop_region clamps at image edge")

    print("\n=== All 7 tests passed ===")


if __name__ == "__main__":
    _self_test()
