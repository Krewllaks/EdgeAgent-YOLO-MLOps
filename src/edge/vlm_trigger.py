"""VLM Async Trigger - PaliGemma activation logic.

When YOLO detection confidence falls below a threshold, this module
queues frames for VLM (PaliGemma) reasoning.  The architecture is
"YOLO uyanirsa VLM calisir" (VLM fires only on uncertain detections).

Usage:
    python src/edge/vlm_trigger.py \
        --model models/phase1_final_ca.pt \
        --source data/processed/phase1_multiclass_v1/test/images \
        --conf-threshold 0.40
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from queue import Queue, Full

ROOT = Path(__file__).resolve().parents[2]


def _register_custom_modules() -> None:
    sys.path.insert(0, str(ROOT))
    from src.models.coordatt import CoordAtt, HSigmoid, HSwish, register_coordatt

    register_coordatt()
    import __main__
    for cls in (HSigmoid, HSwish, CoordAtt):
        setattr(__main__, cls.__name__, cls)


# ── Configuration ────────────────────────────────────────────────────

@dataclass
class TriggerConfig:
    """Defines when the VLM should be activated."""
    conf_threshold: float = 0.40
    target_classes: list[str] = field(
        default_factory=lambda: ["missing_screw", "missing_component"]
    )
    queue_maxsize: int = 32
    vlm_model_id: str = "google/paligemma-3b-mix-224"
    vlm_prompt: str = "Bu goruntudeki vidanin durumunu analiz et. Kusur var mi, neden?"


@dataclass
class TriggerEvent:
    """Single VLM trigger event record."""
    timestamp: str
    image_path: str
    trigger_reason: str
    low_conf_detections: list[dict]
    vlm_response: str | None = None


# ── Async VLM Queue (simulated) ─────────────────────────────────────

class VLMWorker:
    """Background worker that would run PaliGemma inference.

    In Phase 2, this will load the actual VLM model. Currently it
    simulates the reasoning step and records events.
    """

    def __init__(self, config: TriggerConfig, output_path: Path):
        self.config = config
        self.output_path = output_path
        self.queue: Queue[TriggerEvent] = Queue(maxsize=config.queue_maxsize)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.processed = 0
        self.dropped = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def submit(self, event: TriggerEvent) -> bool:
        try:
            self.queue.put_nowait(event)
            return True
        except Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as f:
            while not self._stop.is_set():
                try:
                    event = self.queue.get(timeout=0.5)
                except Exception:
                    continue
                # Phase 2: Replace with actual PaliGemma inference
                event.vlm_response = (
                    f"[SIMULATED] PaliGemma analiz sonucu: "
                    f"{len(event.low_conf_detections)} dusuk guvenli tespit incelendi. "
                    f"Detayli yorum Phase 2'de aktif olacak."
                )
                f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
                f.flush()
                self.processed += 1


# ── Trigger Logic ────────────────────────────────────────────────────

def should_trigger_vlm(
    results,
    class_names: dict[int, str],
    config: TriggerConfig,
) -> tuple[bool, list[dict], str]:
    """Decide whether a frame should be sent to VLM.

    Returns (should_trigger, low_conf_dets, reason).

    Trigger conditions:
    1. No detections at all (potential missed defect)
    2. Any detection with conf < threshold in target classes
    """
    boxes = results[0].boxes if results else None
    if boxes is None or len(boxes) == 0:
        return True, [], "no_detection"

    low_conf = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        if conf < config.conf_threshold and cls_name in config.target_classes:
            low_conf.append({
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(conf, 4),
                "bbox": boxes.xyxy[i].tolist(),
            })

    if low_conf:
        return True, low_conf, "low_confidence"
    return False, [], ""


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM trigger simulation")
    p.add_argument("--model", type=Path, default=ROOT / "models" / "phase1_final_ca.pt")
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "processed" / "phase1_multiclass_v1" / "test" / "images",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--conf-threshold", type=float, default=0.40)
    p.add_argument("--max-events", type=int, default=50)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "generated",
    )
    return p.parse_args()


def main() -> None:
    _register_custom_modules()
    args = parse_args()

    import torch
    from ultralytics import YOLO

    if not args.model.exists():
        sys.exit(f"[ERR] Model not found: {args.model}")
    if not args.source.exists():
        sys.exit(f"[ERR] Source not found: {args.source}")

    device = args.device if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.model))

    # Get class names from model
    class_names: dict[int, str] = {}
    names = getattr(model.model, "names", None) or getattr(model, "names", {})
    if names:
        class_names = {int(k): v for k, v in names.items()}

    config = TriggerConfig(conf_threshold=args.conf_threshold)

    events_path = args.output / "vlm_trigger_events.jsonl"
    # Clear previous events file
    events_path.parent.mkdir(parents=True, exist_ok=True)
    if events_path.exists():
        events_path.unlink()

    worker = VLMWorker(config, events_path)
    worker.start()

    # Scan images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in args.source.iterdir() if p.suffix.lower() in exts])

    triggered_count = 0
    for img in images:
        results = model.predict(str(img), imgsz=args.imgsz, device=device, verbose=False)
        should_fire, low_dets, reason = should_trigger_vlm(results, class_names, config)

        if should_fire and triggered_count < args.max_events:
            event = TriggerEvent(
                timestamp=datetime.now().isoformat(timespec="milliseconds"),
                image_path=str(img.name),
                trigger_reason=reason,
                low_conf_detections=low_dets,
            )
            if worker.submit(event):
                triggered_count += 1

    # Let worker drain the queue
    time.sleep(2.0)
    worker.stop()

    # Summary
    summary = {
        "total_images": len(images),
        "triggered_events": triggered_count,
        "processed_by_vlm": worker.processed,
        "dropped": worker.dropped,
        "conf_threshold": config.conf_threshold,
        "target_classes": config.target_classes,
    }
    summary_path = args.output / "vlm_trigger_events.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[OK] VLM trigger run complete"
        f" - scanned images: {len(images)}"
        f" - triggered events: {triggered_count}"
        f" - dropped events: {worker.dropped}"
        f" - output events: {events_path}"
        f" - summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
