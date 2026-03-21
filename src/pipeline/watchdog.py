"""
Watchdog — Sistem sagligi izleme ve otomatik kurtarma.

Izlenen metrikler:
  - Inference dongusu heartbeat
  - GPU bellek ve sicaklik
  - Disk alani
  - Kamera baglantisi
  - MQTT baglantisi

Kullanim:
    from src.pipeline.watchdog import Watchdog

    watchdog = Watchdog(pipeline)
    watchdog.start()
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Sistem saglik durumu."""
    timestamp: str
    healthy: bool
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_temperature_c: float = 0.0
    disk_free_gb: float = 0.0
    camera_connected: bool = False
    mqtt_connected: bool = False
    inference_fps: float = 0.0
    total_frames: int = 0
    uptime_seconds: float = 0.0
    alerts: list = None

    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []


class Watchdog:
    """Sistem sagligi izleme servisi."""

    def __init__(self, pipeline=None, check_interval_sec: float = 30.0,
                 heartbeat_timeout_sec: float = 10.0,
                 min_disk_gb: float = 1.0,
                 max_gpu_temp_c: float = 85.0,
                 max_gpu_memory_pct: float = 90.0):
        self.pipeline = pipeline
        self.check_interval = check_interval_sec
        self.heartbeat_timeout = heartbeat_timeout_sec
        self.min_disk_gb = min_disk_gb
        self.max_gpu_temp = max_gpu_temp_c
        self.max_gpu_memory_pct = max_gpu_memory_pct

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_status: Optional[HealthStatus] = None
        self._alert_callbacks: list = []
        self._last_frame_count = 0
        self._last_check_time = 0.0

    def _check_gpu(self) -> tuple[float, float, float]:
        """GPU bellek ve sicaklik kontrol."""
        try:
            import torch
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1024**2
                mem_total = torch.cuda.get_device_properties(0).total_mem / 1024**2
                # Sicaklik (nvidia-smi uzerinden)
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=temperature.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5,
                    )
                    temp = float(result.stdout.strip().split("\n")[0])
                except Exception:
                    temp = 0.0
                return mem_used, mem_total, temp
        except ImportError:
            pass
        return 0.0, 0.0, 0.0

    def _check_disk(self) -> float:
        """Disk bos alan (GB)."""
        import shutil
        total, used, free = shutil.disk_usage(Path.cwd())
        return free / (1024**3)

    def _check_health(self) -> HealthStatus:
        """Tam saglik kontrolu."""
        alerts = []

        # GPU
        gpu_used, gpu_total, gpu_temp = self._check_gpu()
        if gpu_total > 0 and (gpu_used / gpu_total * 100) > self.max_gpu_memory_pct:
            alerts.append(f"GPU bellek yuksek: {gpu_used:.0f}/{gpu_total:.0f} MB")
        if gpu_temp > self.max_gpu_temp:
            alerts.append(f"GPU sicaklik yuksek: {gpu_temp:.0f}C (limit: {self.max_gpu_temp}C)")

        # Disk
        disk_free = self._check_disk()
        if disk_free < self.min_disk_gb:
            alerts.append(f"Disk alani dusuk: {disk_free:.1f} GB (min: {self.min_disk_gb} GB)")

        # Pipeline stats
        camera_ok = False
        mqtt_ok = False
        fps = 0.0
        total_frames = 0
        uptime = 0.0

        if self.pipeline:
            stats = self.pipeline.get_stats()
            total_frames = stats.total_frames
            uptime = stats.uptime_seconds

            # FPS hesapla
            now = time.time()
            dt = now - self._last_check_time if self._last_check_time > 0 else self.check_interval
            frame_diff = total_frames - self._last_frame_count
            fps = frame_diff / dt if dt > 0 else 0
            self._last_frame_count = total_frames
            self._last_check_time = now

            # Heartbeat kontrolu
            if uptime > self.heartbeat_timeout and frame_diff == 0 and total_frames > 0:
                alerts.append(f"Inference durmus: {self.heartbeat_timeout:.0f}s icerisinde yeni kare yok")

            # Kamera
            if hasattr(self.pipeline, "_camera") and self.pipeline._camera:
                camera_ok = self.pipeline._camera.is_connected

            # MQTT
            if hasattr(self.pipeline, "_mqtt") and self.pipeline._mqtt:
                mqtt_ok = True  # Bagliysa true

        status = HealthStatus(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            healthy=len(alerts) == 0,
            gpu_memory_used_mb=gpu_used,
            gpu_memory_total_mb=gpu_total,
            gpu_temperature_c=gpu_temp,
            disk_free_gb=disk_free,
            camera_connected=camera_ok,
            mqtt_connected=mqtt_ok,
            inference_fps=round(fps, 1),
            total_frames=total_frames,
            uptime_seconds=round(uptime, 1),
            alerts=alerts,
        )

        # Alarm callback'leri
        if alerts:
            for cb in self._alert_callbacks:
                try:
                    cb(status)
                except Exception:
                    pass

        return status

    def _run_loop(self):
        """Izleme dongusu."""
        logger.info(f"Watchdog baslatildi (kontrol araligi: {self.check_interval}s)")
        while not self._stop_event.is_set():
            try:
                self._last_status = self._check_health()
                if self._last_status.alerts:
                    for alert in self._last_status.alerts:
                        logger.warning(f"[WATCHDOG] {alert}")
                else:
                    logger.debug(f"[WATCHDOG] Saglikli — FPS: {self._last_status.inference_fps}")

                # MQTT'ye saglik durumu yayinla
                if self.pipeline and hasattr(self.pipeline, "_mqtt") and self.pipeline._mqtt:
                    try:
                        import json
                        from dataclasses import asdict
                        self.pipeline._mqtt.publish(
                            "edgeagent/factory/health",
                            json.dumps(asdict(self._last_status), default=str),
                        )
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Watchdog kontrol hatasi: {e}")

            self._stop_event.wait(self.check_interval)

    def start(self):
        """Watchdog'u baslat."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Watchdog'u durdur."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_status(self) -> Optional[HealthStatus]:
        """Son saglik durumunu getir."""
        return self._last_status

    def on_alert(self, callback):
        """Alarm durumunda cagrilacak callback."""
        self._alert_callbacks.append(callback)
