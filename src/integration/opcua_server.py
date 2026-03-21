"""
OPC-UA Sunucusu — Endustri 4.0 SCADA/MES entegrasyonu.

IEC 62541 standartina uyumlu OPC-UA sunucusu.
SCADA sistemleri (Siemens WinCC, TIA Portal vb.) dogrudan baglanabilir.

Yayinlanan degiskenler:
  - LastVerdict (OK/NOK)
  - Confidence (0-1)
  - TotalInspected
  - OKCount / NOKCount
  - QualityRate (%)
  - InferenceLatencyMs
  - ModelVersion
  - SystemHealthy (bool)

Kullanim:
    from src.integration.opcua_server import EdgeAgentOPCUA

    server = EdgeAgentOPCUA(port=4840)
    server.start()
    server.update_verdict("OK", confidence=0.95)
    server.stop()
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OPCUAState:
    """OPC-UA ile paylasilan durum degiskenleri."""
    last_verdict: str = "OK"
    confidence: float = 0.0
    total_inspected: int = 0
    ok_count: int = 0
    nok_count: int = 0
    quality_rate: float = 100.0
    inference_ms: float = 0.0
    model_version: str = ""
    system_healthy: bool = True
    current_shift: str = ""
    lot_number: str = ""


class EdgeAgentOPCUA:
    """OPC-UA sunucusu — SCADA/MES entegrasyonu."""

    def __init__(self, endpoint: str = "opc.tcp://0.0.0.0:4840/edgeagent",
                 server_name: str = "EdgeAgent QC Server"):
        self.endpoint = endpoint
        self.server_name = server_name
        self._server = None
        self._nodes = {}
        self._state = OPCUAState()
        self._lock = threading.Lock()
        self._available = False

        # asyncua kutuphanesi var mi kontrol et
        try:
            from asyncua.sync import Server
            self._available = True
        except ImportError:
            logger.warning(
                "OPC-UA kutuphanesi bulunamadi. Kurmak icin: pip install asyncua"
            )

    @property
    def available(self) -> bool:
        return self._available

    def start(self):
        """OPC-UA sunucusunu baslat."""
        if not self._available:
            logger.warning("OPC-UA baslatilamadi: asyncua kutuphanesi eksik")
            return

        try:
            from asyncua.sync import Server
            from asyncua import ua

            self._server = Server()
            self._server.set_endpoint(self.endpoint)
            self._server.set_server_name(self.server_name)

            # Namespace olustur
            uri = "urn:edgeagent:qc"
            idx = self._server.register_namespace(uri)

            # Degisken dugumleri olustur
            obj = self._server.nodes.objects.add_object(idx, "QualityControl")

            self._nodes["last_verdict"] = obj.add_variable(
                idx, "LastVerdict", "OK", ua.VariantType.String
            )
            self._nodes["confidence"] = obj.add_variable(
                idx, "Confidence", 0.0, ua.VariantType.Float
            )
            self._nodes["total_inspected"] = obj.add_variable(
                idx, "TotalInspected", 0, ua.VariantType.Int32
            )
            self._nodes["ok_count"] = obj.add_variable(
                idx, "OKCount", 0, ua.VariantType.Int32
            )
            self._nodes["nok_count"] = obj.add_variable(
                idx, "NOKCount", 0, ua.VariantType.Int32
            )
            self._nodes["quality_rate"] = obj.add_variable(
                idx, "QualityRate", 100.0, ua.VariantType.Float
            )
            self._nodes["inference_ms"] = obj.add_variable(
                idx, "InferenceLatencyMs", 0.0, ua.VariantType.Float
            )
            self._nodes["model_version"] = obj.add_variable(
                idx, "ModelVersion", "", ua.VariantType.String
            )
            self._nodes["system_healthy"] = obj.add_variable(
                idx, "SystemHealthy", True, ua.VariantType.Boolean
            )
            self._nodes["current_shift"] = obj.add_variable(
                idx, "CurrentShift", "", ua.VariantType.String
            )
            self._nodes["lot_number"] = obj.add_variable(
                idx, "LotNumber", "", ua.VariantType.String
            )

            # Tum degiskenleri okunabilir yap
            for node in self._nodes.values():
                node.set_writable()

            self._server.start()
            logger.info(f"OPC-UA sunucusu baslatildi: {self.endpoint}")
        except Exception as e:
            logger.error(f"OPC-UA baslatma hatasi: {e}")
            self._available = False

    def stop(self):
        """Sunucuyu durdur."""
        if self._server:
            try:
                self._server.stop()
                logger.info("OPC-UA sunucusu durduruldu")
            except Exception:
                pass

    def update_verdict(self, verdict: str, confidence: float = 0.0,
                       inference_ms: float = 0.0):
        """Yeni karar ile degiskenleri guncelle."""
        with self._lock:
            self._state.last_verdict = verdict
            self._state.confidence = confidence
            self._state.inference_ms = inference_ms
            self._state.total_inspected += 1
            if verdict == "OK":
                self._state.ok_count += 1
            else:
                self._state.nok_count += 1
            total = self._state.total_inspected
            self._state.quality_rate = (
                self._state.ok_count / total * 100 if total > 0 else 100
            )

            # OPC-UA dugumleri guncelle
            if self._server and self._nodes:
                try:
                    self._nodes["last_verdict"].write_value(verdict)
                    self._nodes["confidence"].write_value(confidence)
                    self._nodes["total_inspected"].write_value(total)
                    self._nodes["ok_count"].write_value(self._state.ok_count)
                    self._nodes["nok_count"].write_value(self._state.nok_count)
                    self._nodes["quality_rate"].write_value(self._state.quality_rate)
                    self._nodes["inference_ms"].write_value(inference_ms)
                except Exception as e:
                    logger.debug(f"OPC-UA guncelleme hatasi: {e}")

    def update_system_health(self, healthy: bool):
        """Sistem saglik durumunu guncelle."""
        with self._lock:
            self._state.system_healthy = healthy
            if self._server and "system_healthy" in self._nodes:
                try:
                    self._nodes["system_healthy"].write_value(healthy)
                except Exception:
                    pass

    def update_shift(self, shift_name: str):
        """Mevcut vardiya bilgisini guncelle."""
        with self._lock:
            self._state.current_shift = shift_name
            if self._server and "current_shift" in self._nodes:
                try:
                    self._nodes["current_shift"].write_value(shift_name)
                except Exception:
                    pass

    def update_lot(self, lot_number: str):
        """Lot numarasini guncelle."""
        with self._lock:
            self._state.lot_number = lot_number
            if self._server and "lot_number" in self._nodes:
                try:
                    self._nodes["lot_number"].write_value(lot_number)
                except Exception:
                    pass

    def get_state(self) -> OPCUAState:
        return self._state
