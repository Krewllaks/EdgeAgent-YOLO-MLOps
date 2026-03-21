"""
Camera Abstraction Layer — Donanim bagimsiz kamera arayuzu.

Desteklenen kamera tipleri:
  - GigE Vision (GenICam / Basler pypylon)
  - USB kamera (OpenCV VideoCapture)
  - RTSP/IP kamera (GStreamer/ffmpeg)
  - Dosya modu (test icin goruntu/video)
  - Mock modu (unit test)

Kullanim:
    from src.camera.capture import create_camera

    camera = create_camera("usb", device_id=0)
    camera = create_camera("rtsp", url="rtsp://192.168.1.100:554/stream")
    camera = create_camera("file", path="test_video.mp4")
    camera = create_camera("gige")  # otomatik kesfet

    frame = camera.grab_frame()
    camera.release()
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Kamera yapilandirmasi."""
    camera_type: str = "usb"           # usb | rtsp | gige | file | mock
    device_id: int = 0                 # USB kamera ID
    url: str = ""                      # RTSP URL
    file_path: str = ""                # Dosya/video yolu
    width: int = 1280
    height: int = 1024
    fps: int = 30
    exposure_us: int = -1              # -1 = otomatik
    gain: float = -1.0                 # -1 = otomatik
    trigger_mode: str = "freerun"      # freerun | software | hardware
    reconnect_attempts: int = 5
    reconnect_delay_sec: float = 2.0
    mock_image_size: tuple = (640, 640, 3)


class CameraSource(ABC):
    """Soyut kamera arayuzu — tum kamera tipleri bunu uygular."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._connected = False
        self._frame_count = 0
        self._last_frame_time = 0.0

    @abstractmethod
    def connect(self) -> bool:
        """Kameraya baglan. Basarili ise True doner."""

    @abstractmethod
    def grab_frame(self) -> Optional[np.ndarray]:
        """Tek kare yakala. BGR numpy array doner, basarisiz ise None."""

    @abstractmethod
    def release(self):
        """Kamera kaynaklarini serbest birak."""

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps_actual(self) -> float:
        if self._last_frame_time <= 0:
            return 0.0
        elapsed = time.time() - self._last_frame_time
        return 1.0 / elapsed if elapsed > 0 else 0.0

    def reconnect(self) -> bool:
        """Baglanti kopmasinda otomatik yeniden baglanma."""
        for attempt in range(1, self.config.reconnect_attempts + 1):
            logger.warning(f"Yeniden baglanma denemesi {attempt}/{self.config.reconnect_attempts}")
            self.release()
            time.sleep(self.config.reconnect_delay_sec)
            if self.connect():
                logger.info("Yeniden baglanti basarili")
                return True
        logger.error("Yeniden baglanti basarisiz")
        return False


class USBCamera(CameraSource):
    """OpenCV VideoCapture ile USB kamera."""

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._cap = None

    def connect(self) -> bool:
        try:
            self._cap = cv2.VideoCapture(self.config.device_id)
            if not self._cap.isOpened():
                logger.error(f"USB kamera {self.config.device_id} acilamadi")
                return False
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self._connected = True
            logger.info(f"USB kamera {self.config.device_id} baglandi")
            return True
        except Exception as e:
            logger.error(f"USB kamera baglanti hatasi: {e}")
            return False

    def grab_frame(self) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            if not self.reconnect():
                return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Kare yakalanamadi")
            return None
        self._frame_count += 1
        self._last_frame_time = time.time()
        return frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False


class RTSPCamera(CameraSource):
    """RTSP/IP kamera stream."""

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._cap = None

    def connect(self) -> bool:
        try:
            # GStreamer pipeline tercih edilir (daha kararli)
            gst_pipeline = (
                f"rtspsrc location={self.config.url} latency=100 ! "
                f"rtph264depay ! h264parse ! avdec_h264 ! "
                f"videoconvert ! appsink"
            )
            self._cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if not self._cap.isOpened():
                # Fallback: direkt OpenCV
                logger.info("GStreamer basarisiz, direkt RTSP deneniyor...")
                self._cap = cv2.VideoCapture(self.config.url)
            if not self._cap.isOpened():
                logger.error(f"RTSP baglanti basarisiz: {self.config.url}")
                return False
            self._connected = True
            logger.info(f"RTSP kamera baglandi: {self.config.url}")
            return True
        except Exception as e:
            logger.error(f"RTSP baglanti hatasi: {e}")
            return False

    def grab_frame(self) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            if not self.reconnect():
                return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        self._frame_count += 1
        self._last_frame_time = time.time()
        return frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False


class GigECamera(CameraSource):
    """GigE Vision kamera (GenICam / Basler pypylon)."""

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._camera = None
        self._grabber = None

    def connect(self) -> bool:
        # Oncelik 1: Basler pypylon
        try:
            from pypylon import pylon
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            if not devices:
                logger.error("GigE kamera bulunamadi (pypylon)")
                return False
            self._camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            self._camera.Open()
            self._camera.Width.Value = min(self.config.width, self._camera.Width.Max)
            self._camera.Height.Value = min(self.config.height, self._camera.Height.Max)
            if self.config.exposure_us > 0:
                self._camera.ExposureTime.Value = self.config.exposure_us
            if self.config.trigger_mode == "hardware":
                self._camera.TriggerMode.Value = "On"
                self._camera.TriggerSource.Value = "Line1"
            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._connected = True
            logger.info(f"GigE kamera baglandi (pypylon): {devices[0].GetFriendlyName()}")
            return True
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"pypylon hatasi: {e}")

        # Oncelik 2: Harvester (GenICam)
        try:
            from harvesters.core import Harvester
            self._grabber = Harvester()
            # GenTL producer'lari ara
            import glob
            producers = glob.glob("/opt/mvIMPACT_Acquire/lib/**/*.cti", recursive=True)
            producers += glob.glob("C:/Program Files/MATRIX VISION/**/*.cti", recursive=True)
            for p in producers:
                self._grabber.add_file(p)
            self._grabber.update()
            if not self._grabber.device_info_list:
                logger.error("GigE kamera bulunamadi (Harvester)")
                return False
            ia = self._grabber.create()
            ia.start()
            self._camera = ia
            self._connected = True
            logger.info("GigE kamera baglandi (Harvester/GenICam)")
            return True
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Harvester hatasi: {e}")

        logger.error("GigE kamera kutuphanesi bulunamadi (pypylon veya harvesters gerekli)")
        return False

    def grab_frame(self) -> Optional[np.ndarray]:
        if not self._connected:
            return None
        try:
            # pypylon
            if hasattr(self._camera, "RetrieveResult"):
                result = self._camera.RetrieveResult(5000)
                if result.GrabSucceeded():
                    frame = result.Array.copy()
                    result.Release()
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    self._frame_count += 1
                    self._last_frame_time = time.time()
                    return frame
                result.Release()
                return None
            # Harvester
            if hasattr(self._camera, "fetch"):
                with self._camera.fetch() as buffer:
                    payload = buffer.payload
                    comp = payload.components[0]
                    frame = comp.data.reshape(comp.height, comp.width, -1).copy()
                    if frame.shape[2] == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    self._frame_count += 1
                    self._last_frame_time = time.time()
                    return frame
        except Exception as e:
            logger.warning(f"GigE kare yakalama hatasi: {e}")
        return None

    def release(self):
        try:
            if self._camera is not None:
                if hasattr(self._camera, "StopGrabbing"):
                    self._camera.StopGrabbing()
                    self._camera.Close()
                elif hasattr(self._camera, "stop"):
                    self._camera.stop()
                    self._camera.destroy()
            if self._grabber is not None:
                self._grabber.reset()
        except Exception:
            pass
        self._camera = None
        self._grabber = None
        self._connected = False


class FileCamera(CameraSource):
    """Dosya/video tabanlı kamera (test icin)."""

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._cap = None
        self._image_paths = []
        self._image_idx = 0

    def connect(self) -> bool:
        path = Path(self.config.file_path)
        if not path.exists():
            logger.error(f"Dosya bulunamadi: {path}")
            return False

        if path.is_dir():
            # Klasordeki goruntuler
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            self._image_paths = sorted([
                p for p in path.iterdir() if p.suffix.lower() in exts
            ])
            if not self._image_paths:
                logger.error(f"Klasorde goruntu bulunamadi: {path}")
                return False
            self._connected = True
            logger.info(f"Dosya kamera: {len(self._image_paths)} goruntu yuklu")
            return True
        elif path.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
            self._cap = cv2.VideoCapture(str(path))
            if not self._cap.isOpened():
                logger.error(f"Video acilamadi: {path}")
                return False
            self._connected = True
            logger.info(f"Video kamera: {path.name}")
            return True
        else:
            # Tek goruntu
            self._image_paths = [path]
            self._connected = True
            return True

    def grab_frame(self) -> Optional[np.ndarray]:
        if self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Basa sar
                ret, frame = self._cap.read()
                if not ret:
                    return None
            self._frame_count += 1
            self._last_frame_time = time.time()
            return frame
        elif self._image_paths:
            if self._image_idx >= len(self._image_paths):
                self._image_idx = 0  # Basa sar
            path = self._image_paths[self._image_idx]
            self._image_idx += 1
            frame = cv2.imread(str(path))
            if frame is not None:
                self._frame_count += 1
                self._last_frame_time = time.time()
            return frame
        return None

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False


class MockCamera(CameraSource):
    """Unit test icin sahte kamera."""

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._frames = []

    def connect(self) -> bool:
        self._connected = True
        return True

    def add_frame(self, frame: np.ndarray):
        """Test icin kare ekle."""
        self._frames.append(frame)

    def grab_frame(self) -> Optional[np.ndarray]:
        if self._frames:
            frame = self._frames[self._frame_count % len(self._frames)]
        else:
            h, w = self.config.mock_image_size[:2]
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        self._frame_count += 1
        self._last_frame_time = time.time()
        return frame

    def release(self):
        self._connected = False
        self._frames.clear()


# ── Factory Function ──────────────────────────────────────────

CAMERA_TYPES = {
    "usb": USBCamera,
    "rtsp": RTSPCamera,
    "gige": GigECamera,
    "file": FileCamera,
    "mock": MockCamera,
}


def create_camera(camera_type: str = "usb", **kwargs) -> CameraSource:
    """Kamera olustur.

    Args:
        camera_type: "usb", "rtsp", "gige", "file", "mock"
        **kwargs: CameraConfig alanlari

    Returns:
        CameraSource instance (henuz baglanmamis, connect() cagir)
    """
    cls = CAMERA_TYPES.get(camera_type)
    if cls is None:
        raise ValueError(f"Bilinmeyen kamera tipi: {camera_type}. Desteklenen: {list(CAMERA_TYPES)}")

    config = CameraConfig(camera_type=camera_type, **kwargs)
    return cls(config)


def create_camera_from_config(config_path: str) -> CameraSource:
    """YAML config dosyasindan kamera olustur."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cam_cfg = cfg.get("camera", {})
    camera_type = cam_cfg.pop("type", "usb")
    return create_camera(camera_type, **cam_cfg)
