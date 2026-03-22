"""
Donanim Bagimsiz Model Runner — ONNX / TensorRT / PyTorch.

Otomatik secim:
  1. TensorRT .engine varsa → TensorRT (maks performans)
  2. ONNX .onnx varsa → ONNX Runtime (platform bagimsiz)
  3. PyTorch .pt varsa → Ultralytics (gelistirme)

Kullanim:
    from src.pipeline.model_runner import create_model_runner

    runner = create_model_runner("models/phase1_final_ca.pt")
    detections = runner.predict(frame, conf=0.25, imgsz=640)
    runner.release()
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Tek bir tespit sonucu."""
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple  # (x1, y1, x2, y2) piksel
    bbox_xywhn: tuple  # (cx, cy, w, h) normalize


@dataclass
class PredictionResult:
    """Inference sonucu."""
    detections: list[Detection]
    inference_ms: float
    image_shape: tuple  # (h, w)
    model_id: str


from src.common.constants import CLASS_NAMES


class ModelRunner(ABC):
    """Soyut model arayuzu."""

    def __init__(self, model_path: str):
        self.model_path = str(model_path)
        self.model_id = Path(model_path).stem
        self._loaded = False

    @abstractmethod
    def load(self) -> bool:
        """Modeli yukle."""

    @abstractmethod
    def predict(self, frame: np.ndarray, conf: float = 0.25,
                imgsz: int = 640) -> PredictionResult:
        """Inference calistir."""

    @abstractmethod
    def release(self):
        """Model kaynaklarini serbest birak."""

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class UltralyticsRunner(ModelRunner):
    """Ultralytics YOLO ile PyTorch inference."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._model = None

    def load(self) -> bool:
        try:
            from src.models.coordatt import register_coordatt
            register_coordatt()

            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._loaded = True
            logger.info(f"Ultralytics model yuklendi: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Model yukleme hatasi: {e}")
            return False

    def predict(self, frame: np.ndarray, conf: float = 0.25,
                imgsz: int = 640) -> PredictionResult:
        t0 = time.perf_counter()
        results = self._model(frame, conf=conf, imgsz=imgsz, verbose=False)
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = []
        if results and len(results) > 0:
            r = results[0]
            h, w = frame.shape[:2]
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=CLASS_NAMES.get(cls_id, f"cls_{cls_id}"),
                    confidence=float(box.conf[0]),
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_xywhn=(cx, cy, bw, bh),
                ))

        return PredictionResult(
            detections=detections,
            inference_ms=inference_ms,
            image_shape=(frame.shape[0], frame.shape[1]),
            model_id=self.model_id,
        )

    def release(self):
        self._model = None
        self._loaded = False


class ONNXRunner(ModelRunner):
    """ONNX Runtime ile platform-bagimsiz inference."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._session = None

    def load(self) -> bool:
        try:
            import onnxruntime as ort

            providers = []
            if "TensorrtExecutionProvider" in ort.get_available_providers():
                providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            self._session = ort.InferenceSession(self.model_path, providers=providers)
            active = self._session.get_providers()
            self._loaded = True
            logger.info(f"ONNX model yuklendi: {self.model_path} (providers: {active})")
            return True
        except Exception as e:
            logger.error(f"ONNX model yukleme hatasi: {e}")
            return False

    def _preprocess(self, frame: np.ndarray, imgsz: int) -> np.ndarray:
        """BGR image → NCHW float32 tensor."""
        import cv2
        img = cv2.resize(frame, (imgsz, imgsz))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def _postprocess(self, output: np.ndarray, frame_shape: tuple,
                     imgsz: int, conf: float) -> list[Detection]:
        """YOLO ONNX output → Detection listesi."""
        detections = []
        h, w = frame_shape[:2]

        # YOLOv10 ONNX output format: [batch, num_detections, 6] (x1,y1,x2,y2,conf,cls)
        if output.ndim == 3:
            preds = output[0]
        else:
            preds = output

        for pred in preds:
            if len(pred) < 6:
                continue
            score = float(pred[4])
            if score < conf:
                continue
            cls_id = int(pred[5])
            # Scale from imgsz to original
            x1 = float(pred[0]) * w / imgsz
            y1 = float(pred[1]) * h / imgsz
            x2 = float(pred[2]) * w / imgsz
            y2 = float(pred[3]) * h / imgsz
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            detections.append(Detection(
                class_id=cls_id,
                class_name=CLASS_NAMES.get(cls_id, f"cls_{cls_id}"),
                confidence=score,
                bbox_xyxy=(x1, y1, x2, y2),
                bbox_xywhn=(cx, cy, bw, bh),
            ))
        return detections

    def predict(self, frame: np.ndarray, conf: float = 0.25,
                imgsz: int = 640) -> PredictionResult:
        input_tensor = self._preprocess(frame, imgsz)
        input_name = self._session.get_inputs()[0].name

        t0 = time.perf_counter()
        outputs = self._session.run(None, {input_name: input_tensor})
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = self._postprocess(outputs[0], frame.shape, imgsz, conf)

        return PredictionResult(
            detections=detections,
            inference_ms=inference_ms,
            image_shape=(frame.shape[0], frame.shape[1]),
            model_id=self.model_id,
        )

    def release(self):
        self._session = None
        self._loaded = False


# ── Factory ───────────────────────────────────────────────────

def create_model_runner(model_path: str, prefer: str = "auto") -> ModelRunner:
    """Model runner olustur.

    Args:
        model_path: Model dosya yolu (.pt, .onnx, .engine)
        prefer: "auto" | "onnx" | "ultralytics"
            auto: dosya uzantisina gore otomatik sec

    Returns:
        ModelRunner instance (henuz yuklenmemis, load() cagir)
    """
    path = Path(model_path)

    if prefer == "auto":
        if path.suffix == ".onnx":
            return ONNXRunner(model_path)
        elif path.suffix == ".engine":
            # TensorRT — ONNX Runtime TensorRT EP ile
            onnx_path = path.with_suffix(".onnx")
            if onnx_path.exists():
                return ONNXRunner(str(onnx_path))
            logger.warning(f"TensorRT engine icin ONNX dosyasi bulunamadi: {onnx_path}")
            return ONNXRunner(model_path)
        else:
            return UltralyticsRunner(model_path)
    elif prefer == "onnx":
        return ONNXRunner(model_path)
    else:
        return UltralyticsRunner(model_path)
