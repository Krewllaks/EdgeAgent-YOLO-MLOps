"""VLM Async Trigger - PaliGemma activation logic.

When YOLO detection confidence falls below a threshold, this module
queues frames for VLM (PaliGemma) reasoning.  The architecture is
"YOLO uyanirsa VLM calisir" (VLM fires only on uncertain detections).

Phase 2 upgrade:
- Priority queue (lowest confidence = highest priority)
- Real PaliGemma inference via VLMReasoner (Warm Standby)
- Crop-based analysis (YOLO bbox + 20% padding)
- Fallback to simulation if VLM not loaded

Usage:
    python src/edge/vlm_trigger.py \
        --model models/phase1_final_ca.pt \
        --source data/processed/phase1_multiclass_v1/test/images \
        --conf-threshold 0.40
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from queue import PriorityQueue, Full

import numpy as np

logger = logging.getLogger(__name__)

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
    queue_maxsize: int = 5
    vlm_model_id: str = "google/paligemma-3b-mix-224"
    vlm_prompt: str = "Bu goruntudeki vidanin durumunu analiz et. Kusur var mi, neden?"
    use_priority_queue: bool = True
    crop_padding_ratio: float = 0.20


@dataclass
class TriggerEvent:
    """Single VLM trigger event record."""
    timestamp: str
    image_path: str
    trigger_reason: str
    low_conf_detections: list[dict]
    vlm_response: str | None = None
    # Priority queue support (lower = higher priority)
    priority: float = 1.0
    # Image data for VLM crop (not serialized to JSON)
    _image_data: np.ndarray | None = field(default=None, repr=False)

    def __lt__(self, other: TriggerEvent) -> bool:
        """PriorityQueue ordering: lowest priority value = first out."""
        return self.priority < other.priority

    def to_dict(self) -> dict:
        """Serialize to dict, excluding numpy image data."""
        d = asdict(self)
        d.pop("_image_data", None)
        d.pop("priority", None)
        return d


# ── Async VLM Queue (Priority-based) ─────────────────────────────────

class VLMWorker:
    """Background worker that runs PaliGemma inference on queued events.

    Uses priority queue: lowest confidence events are processed first.
    When queue is full, the lowest-priority (highest confidence) item
    is dropped to make room for new events.
    """

    def __init__(
        self,
        config: TriggerConfig,
        output_path: Path,
        reasoner=None,
    ):
        self.config = config
        self.output_path = output_path
        self._reasoner = reasoner  # VLMReasoner instance or None
        self.queue: PriorityQueue[TriggerEvent] = PriorityQueue(
            maxsize=config.queue_maxsize
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.processed = 0
        self.dropped = 0
        # Track items for priority-based eviction
        self._items: list[TriggerEvent] = []

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10.0)

    def submit(self, event: TriggerEvent) -> bool:
        """Submit event to priority queue.

        If queue is full, drops the lowest-priority (highest confidence)
        item to make room for potentially more important events.
        """
        with self._lock:
            try:
                self.queue.put_nowait(event)
                self._items.append(event)
                return True
            except Full:
                # Find and remove lowest priority (highest .priority value)
                if self._items:
                    worst = max(self._items, key=lambda e: e.priority)
                    if event.priority < worst.priority:
                        # New event is more important - swap
                        self._items.remove(worst)
                        self._items.append(event)
                        # Rebuild queue
                        self._rebuild_queue()
                        self.dropped += 1
                        return True
                # New event is less important than all queued items
                self.dropped += 1
                return False

    def _rebuild_queue(self) -> None:
        """Rebuild PriorityQueue from tracked items."""
        new_q: PriorityQueue[TriggerEvent] = PriorityQueue(
            maxsize=self.config.queue_maxsize
        )
        for item in self._items:
            try:
                new_q.put_nowait(item)
            except Full:
                break
        self.queue = new_q

    def _run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as f:
            while not self._stop.is_set():
                try:
                    event = self.queue.get(timeout=0.5)
                except Exception:
                    continue

                # Remove from tracking list
                with self._lock:
                    if event in self._items:
                        self._items.remove(event)

                # Run VLM inference or simulate
                self._process_event(event)

                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
                f.flush()
                self.processed += 1

    def _process_event(self, event: TriggerEvent) -> None:
        """Run VLM inference on event, or simulate if reasoner not available."""
        if self._reasoner is not None and self._reasoner.is_loaded:
            try:
                # Crop and reason
                if event._image_data is not None and event.low_conf_detections:
                    bbox = event.low_conf_detections[0].get("bbox")
                    if bbox:
                        from src.reasoning.vlm_reasoner import VLMReasoner
                        crop = VLMReasoner.crop_region(
                            event._image_data,
                            tuple(bbox),
                            self.config.crop_padding_ratio,
                        )
                        result = self._reasoner.reason(crop)
                        event.vlm_response = (
                            f"[VLM] defect={result.defect_type}, "
                            f"conf={result.confidence_estimate:.2f}, "
                            f"latency={result.latency_ms:.0f}ms | "
                            f"{result.reasoning}"
                        )
                        return
                elif event._image_data is not None:
                    # No specific bbox - analyze full image
                    result = self._reasoner.reason(event._image_data)
                    event.vlm_response = (
                        f"[VLM] defect={result.defect_type}, "
                        f"conf={result.confidence_estimate:.2f}, "
                        f"latency={result.latency_ms:.0f}ms | "
                        f"{result.reasoning}"
                    )
                    return
            except Exception as e:
                logger.error("VLM inference failed: %s", e)
                event.vlm_response = f"[VLM-ERROR] {e}"
                return

        # Simulation fallback
        event.vlm_response = (
            f"[SIMULATED] PaliGemma analiz sonucu: "
            f"{len(event.low_conf_detections)} dusuk guvenli tespit incelendi. "
            f"Detayli yorum icin VLM modelini yukleyin."
        )


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


def compute_event_priority(low_conf_dets: list[dict]) -> float:
    """Compute priority value for queue ordering.

    Lower value = higher priority. Uses minimum confidence among
    detections (most uncertain = most important to analyze).
    """
    if not low_conf_dets:
        return 0.0  # No detections = highest priority
    min_conf = min(d["confidence"] for d in low_conf_dets)
    return min_conf  # Lower confidence = lower priority value = processed first


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM trigger with priority queue")
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
    p.add_argument("--queue-size", type=int, default=5)
    p.add_argument("--enable-vlm", action="store_true", help="Load real PaliGemma model")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "generated",
    )
    return p.parse_args()


def main() -> None:
    _register_custom_modules()
    args = parse_args()

    import cv2
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

    config = TriggerConfig(
        conf_threshold=args.conf_threshold,
        queue_maxsize=args.queue_size,
    )

    # Optionally load VLM
    reasoner = None
    if args.enable_vlm:
        from src.reasoning.vlm_reasoner import VLMReasoner
        reasoner = VLMReasoner()
        reasoner.load_model()
        print(f"[OK] VLM loaded: {reasoner.model_id}")

    events_path = args.output / "vlm_trigger_events.jsonl"
    # Clear previous events file
    events_path.parent.mkdir(parents=True, exist_ok=True)
    if events_path.exists():
        events_path.unlink()

    worker = VLMWorker(config, events_path, reasoner=reasoner)
    worker.start()

    # Scan images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in args.source.iterdir() if p.suffix.lower() in exts])

    triggered_count = 0
    for img in images:
        results = model.predict(str(img), imgsz=args.imgsz, device=device, verbose=False)
        should_fire, low_dets, reason = should_trigger_vlm(results, class_names, config)

        if should_fire and triggered_count < args.max_events:
            # Read image for VLM crop
            img_data = cv2.imread(str(img))
            if img_data is not None:
                img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

            priority = compute_event_priority(low_dets)
            event = TriggerEvent(
                timestamp=datetime.now().isoformat(timespec="milliseconds"),
                image_path=str(img.name),
                trigger_reason=reason,
                low_conf_detections=low_dets,
                priority=priority,
                _image_data=img_data,
            )
            if worker.submit(event):
                triggered_count += 1

    # Let worker drain the queue
    time.sleep(3.0)
    worker.stop()

    # Summary
    summary = {
        "total_images": len(images),
        "triggered_events": triggered_count,
        "processed_by_vlm": worker.processed,
        "dropped": worker.dropped,
        "conf_threshold": config.conf_threshold,
        "target_classes": config.target_classes,
        "queue_strategy": "priority",
        "queue_maxsize": config.queue_maxsize,
        "vlm_enabled": args.enable_vlm,
    }
    summary_path = args.output / "vlm_trigger_events.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[OK] VLM trigger run complete"
        f" - scanned images: {len(images)}"
        f" - triggered events: {triggered_count}"
        f" - processed by VLM: {worker.processed}"
        f" - dropped events: {worker.dropped}"
        f" - output events: {events_path}"
        f" - summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
