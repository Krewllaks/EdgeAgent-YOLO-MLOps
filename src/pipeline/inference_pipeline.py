"""
End-to-End Inference Pipeline — Kamera'dan karara.

Mevcut modulleri tek surekli dongude birlestiren orkestrator:
  Kamera → YOLO → Spatial Logic → Conflict Resolver → MQTT/OPC-UA

Shadow mode destegi:
  Champion + Challenger model paralel calisir, sadece champion karar verir.

Kullanim:
    from src.pipeline.inference_pipeline import InferencePipeline

    pipeline = InferencePipeline.from_config("configs/production_config.yaml")
    pipeline.start()   # ana donguyu baslat
    pipeline.stop()    # durdur
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.common.config import load_yaml

logger = logging.getLogger(__name__)


@dataclass
class InferenceEvent:
    """Tek bir inference olayinin tam kaydi."""
    timestamp: str
    frame_id: int
    product_id: str
    model_version: str
    inference_ms: float
    detections: list
    spatial_decision: str
    vlm_triggered: bool
    final_verdict: str
    confidence: float
    rca_text: str
    challenger_verdict: Optional[str] = None  # shadow mode


@dataclass
class PipelineStats:
    """Pipeline istatistikleri."""
    total_frames: int = 0
    ok_count: int = 0
    nok_count: int = 0
    vlm_triggered_count: int = 0
    avg_inference_ms: float = 0.0
    uptime_seconds: float = 0.0
    last_verdict: str = ""
    last_verdict_time: str = ""


class InferencePipeline:
    """Ana uretim inference dongusu."""

    def __init__(self, config: dict):
        self.config = config
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stats = PipelineStats()
        self._start_time = 0.0
        self._event_callbacks: list = []
        self._shadow_mode = config.get("shadow_mode", False)

        # Moduller (lazy init)
        self._camera = None
        self._model = None
        self._challenger_model = None
        self._spatial = None
        self._conflict_resolver = None
        self._mqtt = None
        self._uncertain_collector = None
        self._audit_logger = None

    @classmethod
    def from_config(cls, config_path: str) -> "InferencePipeline":
        """YAML config'den pipeline olustur."""
        config = load_yaml(Path(config_path))
        return cls(config)

    def _init_modules(self):
        """Tum modulleri baslat."""
        cam_cfg = self.config.get("camera", {})
        model_cfg = self.config.get("model", {})
        pipeline_cfg = self.config.get("pipeline", {})

        # Kamera
        from src.camera.capture import create_camera
        cam_type = cam_cfg.get("type", "usb")
        self._camera = create_camera(cam_type, **{k: v for k, v in cam_cfg.items() if k != "type"})
        if not self._camera.connect():
            logger.warning("Kamera baglantisi basarisiz — MockCamera'ya fallback")
            from src.camera.capture import MockCamera, CameraConfig
            self._camera = MockCamera(CameraConfig(camera_type="mock"))
            self._camera.connect()
            self._camera_degraded = True
        else:
            self._camera_degraded = False

        # Model (champion)
        from src.pipeline.model_runner import create_model_runner
        model_path = model_cfg.get("path", "models/phase1_final_ca.pt")
        if not Path(model_path).exists():
            logger.warning(f"Model dosyasi bulunamadi: {model_path} — pipeline demo modunda")
            self._model = None
            self._model_degraded = True
        else:
            self._model = create_model_runner(model_path)
            if not self._model.load():
                raise RuntimeError(f"Model yuklenemedi: {model_path}")
            self._model_degraded = False

        # Challenger model (shadow mode)
        challenger_path = model_cfg.get("challenger_path")
        if challenger_path and self._shadow_mode:
            self._challenger_model = create_model_runner(challenger_path)
            self._challenger_model.load()
            logger.info(f"Shadow mode: challenger model yuklendi: {challenger_path}")

        # Spatial Logic
        try:
            from src.reasoning.spatial_logic import SpatialAnalyzer
            product_name = pipeline_cfg.get("product_name")
            self._spatial = SpatialAnalyzer(product_name=product_name)
        except Exception as e:
            logger.warning(f"Spatial Logic yuklenemedi: {e}")

        # Conflict Resolver
        try:
            from src.reasoning.conflict_resolver import ConflictResolver
            self._conflict_resolver = ConflictResolver()
        except Exception as e:
            logger.warning(f"Conflict Resolver yuklenemedi: {e}")

        # MQTT
        mqtt_cfg = self.config.get("mqtt", {})
        if mqtt_cfg.get("enabled", False):
            try:
                from src.edge.mqtt_bridge import MQTTBridge
                self._mqtt = MQTTBridge(
                    broker=mqtt_cfg.get("broker", "localhost"),
                    port=mqtt_cfg.get("port", 1883),
                )
                self._mqtt.connect()
            except Exception as e:
                logger.warning(f"MQTT baglantisi basarisiz: {e}")

        # Uncertain Collector
        try:
            from src.data.uncertain_collector import UncertainCollector
            self._uncertain_collector = UncertainCollector()
        except Exception as e:
            logger.warning(f"UncertainCollector yuklenemedi: {e}")

        # Audit Logger
        try:
            from src.pipeline.audit_logger import AuditLogger
            self._audit_logger = AuditLogger()
        except Exception as e:
            logger.warning(f"AuditLogger yuklenemedi: {e}")

        logger.info("Tum moduller baslatildi")

    def _process_frame(self, frame: np.ndarray, frame_id: int) -> InferenceEvent:
        """Tek kareyi isle: YOLO → Spatial → Conflict → Karar."""
        pipeline_cfg = self.config.get("pipeline", {})
        conf_threshold = pipeline_cfg.get("confidence_threshold", 0.25)
        vlm_threshold = pipeline_cfg.get("vlm_trigger_threshold", 0.40)
        imgsz = pipeline_cfg.get("imgsz", 640)

        # ── YOLO Inference ─────────────────────────────────────
        if self._model is None:
            from src.pipeline.model_runner import PredictionResult
            result = PredictionResult(detections=[], inference_ms=0.0, frame_shape=frame.shape)
        else:
            result = self._model.predict(frame, conf=conf_threshold, imgsz=imgsz)

        # ── Spatial Logic ──────────────────────────────────────
        spatial_decision = "unknown"
        if self._spatial and result.detections:
            try:
                boxes = []
                for d in result.detections:
                    cx, cy, w, h = d.bbox_xywhn
                    boxes.append({
                        "class_id": d.class_id,
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox": [cx, cy, w, h],
                    })
                spatial_result = self._spatial.analyze_frame(boxes)
                spatial_decision = spatial_result.get("decision", "unknown")
            except Exception as e:
                logger.warning("Spatial logic hatasi: %s", e)

        # ── VLM Trigger Check ─────────────────────────────────
        vlm_triggered = False
        min_conf = min((d.confidence for d in result.detections), default=1.0)
        if min_conf < vlm_threshold and self._uncertain_collector:
            vlm_triggered = True
            try:
                self._uncertain_collector.collect_frame(
                    frame, [{"class_id": d.class_id, "confidence": d.confidence,
                             "bbox": list(d.bbox_xywhn)} for d in result.detections]
                )
            except Exception as e:
                logger.warning("UncertainCollector: %s", e)

        # ── Conflict Resolution ───────────────────────────────
        final_verdict = "OK"
        confidence = 1.0
        rca_text = ""

        if self._conflict_resolver and result.detections:
            try:
                yolo_decision = "OK"
                for d in result.detections:
                    if d.class_id in (1, 2):
                        yolo_decision = d.class_name
                        break

                verdict = self._conflict_resolver.resolve(
                    yolo_decision=yolo_decision,
                    spatial_decision=spatial_decision,
                    vlm_decision=None,
                )
                final_verdict = verdict.verdict
                confidence = verdict.confidence
                rca_text = getattr(verdict, "rca_text", "")
            except Exception as e:
                logger.warning("Conflict resolver hatasi: %s", e)
                # Fallback: YOLO'nun kararini kullan
                for d in result.detections:
                    if d.class_id in (1, 2):
                        final_verdict = d.class_name
                        confidence = d.confidence
                        break
        elif result.detections:
            for d in result.detections:
                if d.class_id in (1, 2):
                    final_verdict = d.class_name
                    confidence = d.confidence
                    break

        # ── Shadow Mode (Challenger) ──────────────────────────
        challenger_verdict = None
        if self._challenger_model and self._shadow_mode:
            try:
                ch_result = self._challenger_model.predict(frame, conf=conf_threshold, imgsz=imgsz)
                challenger_verdict = "OK"
                for d in ch_result.detections:
                    if d.class_id in (1, 2):
                        challenger_verdict = d.class_name
                        break
            except Exception as e:
                logger.debug("Shadow mode hatasi: %s", e)

        # ── MQTT Publish ──────────────────────────────────────
        if self._mqtt:
            try:
                import json
                self._mqtt.publish("edgeagent/factory/results", json.dumps({
                    "verdict": final_verdict,
                    "confidence": round(confidence, 4),
                    "frame_id": frame_id,
                    "inference_ms": round(result.inference_ms, 2),
                    "timestamp": datetime.now().isoformat(),
                }))
            except Exception as e:
                logger.debug("MQTT publish hatasi: %s", e)

        # ── Stats ─────────────────────────────────────────────
        self._stats.total_frames += 1
        if final_verdict == "OK":
            self._stats.ok_count += 1
        else:
            self._stats.nok_count += 1
        if vlm_triggered:
            self._stats.vlm_triggered_count += 1
        # Running average
        n = self._stats.total_frames
        self._stats.avg_inference_ms = (
            self._stats.avg_inference_ms * (n - 1) + result.inference_ms
        ) / n
        self._stats.last_verdict = final_verdict
        self._stats.last_verdict_time = datetime.now().isoformat(timespec="seconds")

        event = InferenceEvent(
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            frame_id=frame_id,
            product_id=f"P{frame_id:08d}",
            model_version=result.model_id,
            inference_ms=result.inference_ms,
            detections=[{
                "class": d.class_name, "conf": round(d.confidence, 4),
                "bbox": list(d.bbox_xyxy),
            } for d in result.detections],
            spatial_decision=spatial_decision,
            vlm_triggered=vlm_triggered,
            final_verdict=final_verdict,
            confidence=confidence,
            rca_text=rca_text,
            challenger_verdict=challenger_verdict,
        )

        # ── Audit Log ─────────────────────────────────────────
        if self._audit_logger:
            try:
                self._audit_logger.log_event(event)
            except Exception as e:
                logger.debug("Audit log hatasi: %s", e)

        return event

    def _run_loop(self):
        """Ana inference dongusu (ayri thread'de calisir)."""
        logger.info("Inference dongusu baslatildi")
        self._start_time = time.time()
        frame_id = 0

        while not self._stop_event.is_set():
            frame = self._camera.grab_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            event = self._process_frame(frame, frame_id)
            frame_id += 1

            # Callback'leri cagir
            for cb in self._event_callbacks:
                try:
                    cb(event)
                except Exception as e:
                    logger.debug("Callback hatasi: %s", e)

        logger.info("Inference dongusu durduruldu")

    def start(self):
        """Pipeline'i baslat (non-blocking)."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Pipeline zaten calisiyor")
            return

        self._init_modules()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Pipeline baslatildi")

    def stop(self):
        """Pipeline'i durdur."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._camera:
            self._camera.release()
        if self._model:
            self._model.release()
        if self._challenger_model:
            self._challenger_model.release()
        logger.info("Pipeline durduruldu")

    def process_single(self, frame: np.ndarray) -> InferenceEvent:
        """Tek kare isle (pipeline baslatmadan, test icin)."""
        if not self._model or not self._model.is_loaded:
            self._init_modules()
        return self._process_frame(frame, 0)

    def get_stats(self) -> PipelineStats:
        """Guncel istatistikleri getir."""
        if self._start_time > 0:
            self._stats.uptime_seconds = time.time() - self._start_time
        return self._stats

    def on_event(self, callback):
        """Her inference olayinda cagrilacak callback ekle."""
        self._event_callbacks.append(callback)

    def hot_swap_model(self, new_model_path: str) -> bool:
        """Uretim durmadan model degistir."""
        from src.pipeline.model_runner import create_model_runner
        new_runner = create_model_runner(new_model_path)
        if not new_runner.load():
            logger.error(f"Yeni model yuklenemedi: {new_model_path}")
            return False

        old_model = self._model
        self._model = new_runner
        if old_model:
            old_model.release()
        logger.info(f"Hot-swap tamamlandi: {new_model_path}")
        return True
