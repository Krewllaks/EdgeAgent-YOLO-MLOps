"""
Audit Logger — Kurumsal izlenebilirlik ve uyumluluk.

Her inference olayini degistirilemez (append-only) log'a yazar.
FDA 21 CFR Part 11 / EU GMP Annex 11 uyumlu kayit tutma.

Log formati: JSONL (her satir bir olay)
Saklama: Lokal SQLite + JSONL dosya

Kullanim:
    from src.pipeline.audit_logger import AuditLogger

    logger = AuditLogger()
    logger.log_event(inference_event)
    events = logger.query(start="2026-03-21", end="2026-03-22")
"""

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """Degistirilemez audit log sistemi."""

    def __init__(self, log_dir: str = "data/audit",
                 db_name: str = "audit.db",
                 jsonl_prefix: str = "audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.log_dir / db_name
        self._jsonl_prefix = jsonl_prefix
        self._lock = threading.Lock()
        self._event_count = 0
        self._init_db()

    def _init_db(self):
        """SQLite tablosunu olustur."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    frame_id INTEGER,
                    product_id TEXT,
                    model_version TEXT,
                    inference_ms REAL,
                    final_verdict TEXT,
                    confidence REAL,
                    spatial_decision TEXT,
                    vlm_triggered INTEGER,
                    rca_text TEXT,
                    challenger_verdict TEXT,
                    detections_json TEXT,
                    event_hash TEXT,
                    lot_number TEXT,
                    operator_id TEXT,
                    line_id TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_verdict ON audit_log(final_verdict)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lot ON audit_log(lot_number)
            """)

    def _compute_hash(self, data: dict) -> str:
        """Olay verisinin SHA256 hash'i (kurcalama onleme)."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def log_event(self, event, lot_number: str = "",
                  operator_id: str = "", line_id: str = ""):
        """Inference olayini kaydet."""
        with self._lock:
            # Event'i dict'e cevir
            if hasattr(event, "__dataclass_fields__"):
                data = asdict(event)
            elif isinstance(event, dict):
                data = event
            else:
                data = vars(event)

            # Hash hesapla
            event_hash = self._compute_hash(data)
            data["event_hash"] = event_hash
            data["lot_number"] = lot_number
            data["operator_id"] = operator_id
            data["line_id"] = line_id

            # JSONL dosyasina yaz (append-only)
            today = datetime.now().strftime("%Y%m%d")
            jsonl_path = self.log_dir / f"{self._jsonl_prefix}_{today}.jsonl"
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")

            # SQLite'a yaz
            try:
                detections_json = json.dumps(data.get("detections", []), default=str)
                with sqlite3.connect(str(self._db_path)) as conn:
                    conn.execute("""
                        INSERT INTO audit_log
                        (timestamp, frame_id, product_id, model_version,
                         inference_ms, final_verdict, confidence,
                         spatial_decision, vlm_triggered, rca_text,
                         challenger_verdict, detections_json, event_hash,
                         lot_number, operator_id, line_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data.get("timestamp", ""),
                        data.get("frame_id", 0),
                        data.get("product_id", ""),
                        data.get("model_version", ""),
                        data.get("inference_ms", 0),
                        data.get("final_verdict", ""),
                        data.get("confidence", 0),
                        data.get("spatial_decision", ""),
                        1 if data.get("vlm_triggered") else 0,
                        data.get("rca_text", ""),
                        data.get("challenger_verdict"),
                        detections_json,
                        event_hash,
                        lot_number,
                        operator_id,
                        line_id,
                    ))
            except Exception as e:
                logger.error(f"SQLite yazma hatasi: {e}")

            self._event_count += 1

    def query(self, start: str = "", end: str = "",
              verdict: str = "", lot_number: str = "",
              limit: int = 1000) -> list[dict]:
        """Audit log'dan sorgulama."""
        conditions = []
        params = []

        if start:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end:
            conditions.append("timestamp <= ?")
            params.append(end)
        if verdict:
            conditions.append("final_verdict = ?")
            params.append(verdict)
        if lot_number:
            conditions.append("lot_number = ?")
            params.append(lot_number)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM audit_log WHERE {where} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        results = []
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)
                for row in cursor:
                    results.append(dict(row))
        except Exception as e:
            logger.error(f"Audit sorgu hatasi: {e}")

        return results

    def get_summary(self, date: str = "") -> dict:
        """Belirli bir gun icin ozet istatistikler."""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                row = conn.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN final_verdict = 'OK' THEN 1 ELSE 0 END) as ok_count,
                        SUM(CASE WHEN final_verdict != 'OK' THEN 1 ELSE 0 END) as nok_count,
                        AVG(inference_ms) as avg_latency,
                        AVG(confidence) as avg_confidence,
                        SUM(vlm_triggered) as vlm_count
                    FROM audit_log
                    WHERE date(timestamp) = ?
                """, (date,)).fetchone()

                if row and row[0] > 0:
                    return {
                        "date": date,
                        "total": row[0],
                        "ok_count": row[1],
                        "nok_count": row[2],
                        "quality_rate": round(row[1] / row[0] * 100, 2) if row[0] > 0 else 0,
                        "avg_latency_ms": round(row[3], 2) if row[3] else 0,
                        "avg_confidence": round(row[4], 4) if row[4] else 0,
                        "vlm_trigger_count": row[5] or 0,
                    }
        except Exception as e:
            logger.error(f"Ozet sorgu hatasi: {e}")

        return {"date": date, "total": 0}

    @property
    def event_count(self) -> int:
        return self._event_count
