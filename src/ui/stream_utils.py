"""Frame annotation and MJPEG streaming utilities.

Kareye bbox, sinif etiketi, guven skoru ve verdict banner cizer.
MJPEG generator ile FastAPI StreamingResponse'a besler.
"""

import time
from typing import Optional

import cv2
import numpy as np

# Sinif renkleri (BGR — OpenCV formati)
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 255, 0),    # screw: yesil
    1: (0, 0, 255),    # missing_screw: kirmizi
    2: (0, 165, 255),  # missing_component: turuncu
}

VERDICT_COLORS: dict[str, tuple[int, int, int]] = {
    "OK": (0, 180, 0),         # yesil
    "missing_screw": (0, 0, 220),      # kirmizi
    "missing_component": (0, 100, 255),  # turuncu
}

# "Kamera bekleniyor" placeholder
_PLACEHOLDER: Optional[np.ndarray] = None


def _get_placeholder() -> np.ndarray:
    """Lazy-init placeholder frame."""
    global _PLACEHOLDER
    if _PLACEHOLDER is None:
        _PLACEHOLDER = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            _PLACEHOLDER, "Kamera bekleniyor...", (100, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2,
        )
    return _PLACEHOLDER


def annotate_frame(
    frame: np.ndarray,
    detections: list,
    verdict: str,
) -> np.ndarray:
    """Kareye bbox + etiket + verdict banner cizer.

    Args:
        frame: BGR numpy array (orijinal kare).
        detections: Detection dataclass listesi (bbox_xyxy, class_id, class_name, confidence).
        verdict: "OK" | "missing_screw" | "missing_component"

    Returns:
        Annotated frame (kopya, orijinal degismez).
    """
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        color = CLASS_COLORS.get(det.class_id, (255, 255, 255))
        thickness = 2

        # Bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Etiket arka plani
        label = f"{det.class_name} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    # Verdict banner (ust serit)
    h, w = annotated.shape[:2]
    banner_color = VERDICT_COLORS.get(verdict, (0, 0, 220))
    cv2.rectangle(annotated, (0, 0), (w, 32), banner_color, -1)

    display_verdict = "OK" if verdict == "OK" else f"NOK: {verdict}"
    cv2.putText(
        annotated, display_verdict, (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )

    # Detection sayisi (sag ust)
    count_text = f"{len(detections)} tespit"
    (cw, _), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        annotated, count_text, (w - cw - 10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )

    return annotated


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Frame'i JPEG bytes olarak encode et."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def mjpeg_generator(frame_buffer, fps_limit: int = 15, quality: int = 80):
    """MJPEG stream generator — FastAPI StreamingResponse icin.

    Args:
        frame_buffer: FrameBuffer instance.
        fps_limit: Maksimum FPS (bant genisligi kontrolu).
        quality: JPEG kalitesi (0-100).

    Yields:
        MJPEG frame bytes (multipart boundary ile).
    """
    interval = 1.0 / fps_limit

    while True:
        af = frame_buffer.wait_for_new(timeout=2.0)

        if af is None:
            jpeg = encode_jpeg(_get_placeholder(), quality=50)
        else:
            jpeg = encode_jpeg(af.frame, quality=quality)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )
        time.sleep(interval)
