"""Thread-safe circular frame buffer for live streaming.

Pipeline'dan gelen annotated frame'leri saklar,
MJPEG stream consumer'lara non-blocking olarak sunar.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AnnotatedFrame:
    """Bbox + verdict banner cizilmis kare."""

    frame: np.ndarray = field(repr=False)  # BGR, bbox'lar cizili
    frame_id: int = 0
    timestamp: float = 0.0
    verdict: str = "OK"
    detection_count: int = 0


class FrameBuffer:
    """Thread-safe circular buffer — producer (pipeline) non-blocking yazar,
    consumer (MJPEG stream) event-driven okur.

    Args:
        maxlen: Maksimum frame sayisi (varsayilan 30 = ~1sn @ 30fps).
    """

    def __init__(self, maxlen: int = 30):
        self._buffer: deque[AnnotatedFrame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._event = threading.Event()

    def put(self, frame: AnnotatedFrame) -> None:
        """Yeni frame ekle (non-blocking, deque maxlen tasarsa eski duser)."""
        with self._lock:
            self._buffer.append(frame)
        self._event.set()

    def get_latest(self) -> Optional[AnnotatedFrame]:
        """Son frame'i dondur (None = henuz frame yok)."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def wait_for_new(self, timeout: float = 1.0) -> Optional[AnnotatedFrame]:
        """Yeni frame gelene kadar bekle, sonra dondur."""
        self._event.wait(timeout=timeout)
        self._event.clear()
        return self.get_latest()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def has_frames(self) -> bool:
        with self._lock:
            return len(self._buffer) > 0
