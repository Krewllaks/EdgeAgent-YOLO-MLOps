"""
Vardiya Yonetimi ve Raporlama.

Otomatik vardiya tespiti, vardiya bazli istatistikler ve raporlama.

Varsayilan vardiyalar:
  - Sabah:  08:00 - 16:00
  - Aksam:  16:00 - 24:00
  - Gece:   24:00 - 08:00

Kullanim:
    from src.pipeline.shift_logger import ShiftLogger

    shift_logger = ShiftLogger()
    shift_logger.record_verdict("OK", confidence=0.95)
    report = shift_logger.get_shift_report()
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ShiftConfig:
    """Vardiya yapilandirmasi."""
    shifts: list = field(default_factory=lambda: [
        {"name": "Sabah", "start": "08:00", "end": "16:00"},
        {"name": "Aksam", "start": "16:00", "end": "00:00"},
        {"name": "Gece", "start": "00:00", "end": "08:00"},
    ])


@dataclass
class ShiftStats:
    """Vardiya istatistikleri."""
    shift_name: str
    date: str
    start_time: str
    total_inspected: int = 0
    ok_count: int = 0
    nok_count: int = 0
    nok_breakdown: dict = field(default_factory=dict)
    avg_inference_ms: float = 0.0
    vlm_trigger_count: int = 0
    quality_rate: float = 0.0
    operator_feedback_count: int = 0

    def update_quality_rate(self):
        if self.total_inspected > 0:
            self.quality_rate = round(self.ok_count / self.total_inspected * 100, 2)


class ShiftLogger:
    """Vardiya bazli uretim takibi."""

    def __init__(self, config: Optional[ShiftConfig] = None,
                 log_dir: str = "data/shifts"):
        self.config = config or ShiftConfig()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_stats: Optional[ShiftStats] = None
        self._current_shift_name = ""
        self._inference_times = []

    def _get_current_shift(self) -> str:
        """Mevcut saate gore vardiya ismini dondur."""
        now = datetime.now().time()
        for shift in self.config.shifts:
            start = dtime.fromisoformat(shift["start"])
            end = dtime.fromisoformat(shift["end"])
            if end <= start:
                # Gece gecisi (ornegin 16:00-00:00 veya 00:00-08:00)
                if now >= start or now < end:
                    return shift["name"]
            else:
                if start <= now < end:
                    return shift["name"]
        return "Bilinmeyen"

    def _ensure_shift(self):
        """Gecerli vardiya icin stats olustur veya varsa devam et."""
        current = self._get_current_shift()
        today = datetime.now().strftime("%Y-%m-%d")

        if self._current_stats is None or self._current_shift_name != current:
            # Onceki vardiyayi kaydet
            if self._current_stats and self._current_stats.total_inspected > 0:
                self._save_shift_report(self._current_stats)

            self._current_shift_name = current
            self._current_stats = ShiftStats(
                shift_name=current,
                date=today,
                start_time=datetime.now().isoformat(timespec="seconds"),
            )
            self._inference_times = []
            logger.info(f"Yeni vardiya baslatildi: {current} ({today})")

    def record_verdict(self, verdict: str, confidence: float = 0.0,
                       inference_ms: float = 0.0, vlm_triggered: bool = False):
        """Bir urun kararini kaydet."""
        self._ensure_shift()
        stats = self._current_stats

        stats.total_inspected += 1
        if verdict == "OK":
            stats.ok_count += 1
        else:
            stats.nok_count += 1
            stats.nok_breakdown[verdict] = stats.nok_breakdown.get(verdict, 0) + 1
        if vlm_triggered:
            stats.vlm_trigger_count += 1
        if inference_ms > 0:
            self._inference_times.append(inference_ms)
            stats.avg_inference_ms = sum(self._inference_times) / len(self._inference_times)
        stats.update_quality_rate()

    def get_shift_report(self) -> Optional[ShiftStats]:
        """Mevcut vardiya raporunu getir."""
        self._ensure_shift()
        return self._current_stats

    def _save_shift_report(self, stats: ShiftStats):
        """Vardiya raporunu dosyaya kaydet."""
        filename = f"shift_{stats.date}_{stats.shift_name}.json"
        filepath = self.log_dir / filename
        try:
            data = asdict(stats)
            data["saved_at"] = datetime.now().isoformat(timespec="seconds")
            filepath.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(f"Vardiya raporu kaydedildi: {filepath}")
        except Exception as e:
            logger.error(f"Vardiya raporu kaydetme hatasi: {e}")

    def end_shift(self):
        """Mevcut vardiyayi sonlandir ve kaydet."""
        if self._current_stats and self._current_stats.total_inspected > 0:
            self._save_shift_report(self._current_stats)
        self._current_stats = None
        self._current_shift_name = ""

    def get_daily_summary(self, date: str = "") -> dict:
        """Belirli bir gunun tum vardiya raporlarini getir."""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        summaries = []
        for filepath in sorted(self.log_dir.glob(f"shift_{date}_*.json")):
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                summaries.append(data)
            except Exception:
                continue

        total = sum(s.get("total_inspected", 0) for s in summaries)
        ok = sum(s.get("ok_count", 0) for s in summaries)
        nok = sum(s.get("nok_count", 0) for s in summaries)

        return {
            "date": date,
            "shifts": summaries,
            "total_inspected": total,
            "ok_count": ok,
            "nok_count": nok,
            "quality_rate": round(ok / total * 100, 2) if total > 0 else 0,
            "defects_per_million": round(nok / total * 1_000_000) if total > 0 else 0,
        }
