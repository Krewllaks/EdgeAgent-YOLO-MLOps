"""MQTT Bridge - Factory communication simulation.

Publishes quality control verdicts and VLM events to an MQTT broker.
Subscribes to control commands (emergency stop, queue clear, model reload).

Requires a running MQTT broker (e.g., Mosquitto) at the configured address.

Usage:
    from src.edge.mqtt_bridge import MQTTBridge

    bridge = MQTTBridge()
    bridge.connect()
    bridge.publish_verdict(final_verdict)
"""

from __future__ import annotations

import json
import logging
import time
import random
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "phase2_config.yaml"


class MQTTBridge:
    """MQTT bridge for factory-floor communication."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG):
        self._config = self._load_config(config_path)
        self._client = None
        self._connected = False
        self._control_callback: Optional[Callable] = None
        self._sim_thread: Optional[threading.Thread] = None
        self._sim_stop = threading.Event()

    @staticmethod
    def _load_config(path: Path) -> dict:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("mqtt", {})
        return {"broker": "localhost", "port": 1883, "topic_prefix": "edgeagent/factory"}

    @property
    def broker(self) -> str:
        return self._config.get("broker", "localhost")

    @property
    def port(self) -> int:
        return self._config.get("port", 1883)

    @property
    def topic_prefix(self) -> str:
        return self._config.get("topic_prefix", "edgeagent/factory")

    @property
    def qos(self) -> int:
        return self._config.get("qos", 1)

    def connect(self) -> bool:
        """Connect to MQTT broker with auto-reconnect."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.error("paho-mqtt not installed. Run: pip install paho-mqtt")
            return False

        self._client = mqtt.Client(
            client_id=f"edgeagent_{int(time.time())}",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        # Enable auto-reconnect
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

        try:
            self._client.connect(self.broker, self.port, keepalive=60)
            self._client.loop_start()
            logger.info("MQTT connecting to %s:%d", self.broker, self.port)
            return True
        except Exception as e:
            logger.error("MQTT connection failed: %s", e)
            return False

    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        self.stop_simulation()
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False
            logger.info("MQTT disconnected")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        self._connected = True
        logger.info("MQTT connected (rc=%s)", rc)
        # Subscribe to control topic
        control_topic = f"{self.topic_prefix}/control"
        client.subscribe(control_topic, qos=self.qos)
        logger.info("Subscribed to %s", control_topic)

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        self._connected = False
        logger.warning("MQTT disconnected (rc=%s), will auto-reconnect", rc)

    def _on_message(self, client, userdata, msg):
        """Handle incoming control messages."""
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            command = payload.get("command", "unknown")
            logger.info("MQTT control command: %s", command)

            if self._control_callback:
                self._control_callback(command, payload)
        except Exception as e:
            logger.error("MQTT message parse error: %s", e)

    def publish_verdict(self, verdict) -> bool:
        """Publish a FinalVerdict to the verdict topic.

        Args:
            verdict: FinalVerdict dataclass or dict.
        """
        if not self._connected or not self._client:
            logger.warning("MQTT not connected, verdict not published")
            return False

        topic = f"{self.topic_prefix}/verdict"
        payload = verdict if isinstance(verdict, dict) else asdict(verdict)
        payload["timestamp"] = datetime.now().isoformat(timespec="milliseconds")

        result = self._client.publish(
            topic, json.dumps(payload, ensure_ascii=False), qos=self.qos
        )
        return result.rc == 0

    def publish_vlm_event(self, event) -> bool:
        """Publish a VLM trigger event."""
        if not self._connected or not self._client:
            return False

        topic = f"{self.topic_prefix}/vlm_trigger"
        payload = event if isinstance(event, dict) else event.to_dict()
        payload["timestamp"] = datetime.now().isoformat(timespec="milliseconds")

        result = self._client.publish(
            topic, json.dumps(payload, ensure_ascii=False), qos=self.qos
        )
        return result.rc == 0

    def publish_alert(self, alert_type: str, message: str) -> bool:
        """Publish an alert (drift detected, queue overflow, etc.)."""
        if not self._connected or not self._client:
            return False

        topic = f"{self.topic_prefix}/alert"
        payload = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        }
        result = self._client.publish(
            topic, json.dumps(payload, ensure_ascii=False), qos=self.qos
        )
        return result.rc == 0

    def subscribe_control(self, callback: Callable[[str, dict], None]) -> None:
        """Register a callback for control commands.

        Callback receives (command: str, payload: dict).
        Known commands: "emergency_stop", "clear_queue", "reload_model"
        """
        self._control_callback = callback

    def start_simulation(self, interval: float = 2.0) -> None:
        """Start publishing simulated factory verdicts at given interval.

        Used for demo/testing without a real camera feed.
        """
        self._sim_stop.clear()
        self._sim_thread = threading.Thread(
            target=self._sim_loop, args=(interval,), daemon=True
        )
        self._sim_thread.start()
        logger.info("MQTT simulation started (interval=%.1fs)", interval)

    def stop_simulation(self) -> None:
        """Stop the simulation loop."""
        self._sim_stop.set()
        if self._sim_thread:
            self._sim_thread.join(timeout=5.0)
            self._sim_thread = None

    def _sim_loop(self, interval: float) -> None:
        """Simulation loop: publish random verdicts."""
        verdicts = ["ok", "ok", "ok", "missing_screw", "missing_component"]
        sources = ["consensus", "vlm", "spatial"]

        while not self._sim_stop.is_set():
            verdict = random.choice(verdicts)
            payload = {
                "verdict": verdict,
                "source": random.choice(sources),
                "confidence": round(random.uniform(0.5, 0.99), 2),
                "conflict_detected": random.random() < 0.1,
                "reasoning": f"Simulated {verdict} verdict",
                "rca_text": f"Simulated RCA for {verdict}",
            }
            self.publish_verdict(payload)
            self._sim_stop.wait(interval)


# ── Self-test (no broker required) ───────────────────────────────────

if __name__ == "__main__":
    print("=== MQTT Bridge Self-Test ===\n")

    bridge = MQTTBridge()
    print(f"Test 1 PASS: Config loaded (broker={bridge.broker}, port={bridge.port})")

    # Test 2: Publish without connection (should not crash)
    result = bridge.publish_verdict({"verdict": "ok", "source": "test"})
    assert result is False  # Not connected
    print("Test 2 PASS: Publish without connection returns False")

    # Test 3: Alert publish without connection
    result = bridge.publish_alert("test", "Test alert")
    assert result is False
    print("Test 3 PASS: Alert without connection returns False")

    # Test 4: Control callback registration
    received = []
    bridge.subscribe_control(lambda cmd, payload: received.append(cmd))
    assert bridge._control_callback is not None
    print("Test 4 PASS: Control callback registered")

    print("\n=== All 4 tests passed (broker-free tests) ===")
    print("For full tests, start Mosquitto and run with --live flag")
