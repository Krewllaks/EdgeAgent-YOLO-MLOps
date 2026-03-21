"""
EdgeAgent — Unified Production Entry Point.

Fabrika PC'sinde tek komutla tum sistemi baslatir:
  - Inference Pipeline (kamera → YOLO → karar)
  - FastAPI HMI (operator arayuzu)
  - Watchdog (saglik izleme)
  - Shift Logger (vardiya raporlama)
  - OPC-UA Server (SCADA entegrasyonu)
  - MQTT Bridge (IoT iletisimi)

Kullanim:
    python main.py                             # Varsayilan: inference mode
    python main.py --mode inference             # Uretim (pipeline + HMI)
    python main.py --mode training              # Surekli egitim kontrolu
    python main.py --mode dev                   # Gelistirme (Streamlit)
    python main.py --config configs/custom.yaml # Ozel konfigürasyon

Servis olarak:
    # Windows: deploy/install-service.ps1 ile NSSM servisi
    # Linux:   deploy/edgeagent.service ile systemd daemon
    # Docker:  docker compose up -d
"""

import argparse
import atexit
import json
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Logging Setup ─────────────────────────────────────────────────

def setup_logging(log_dir: Path, level: str = "INFO"):
    """Rotating file + console logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "edgeagent.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler: 10MB, 5 backups
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return logging.getLogger("edgeagent")


# ── PID Management ────────────────────────────────────────────────

def write_pid(pid_path: Path):
    """PID dosyasi yaz — cift baslatmayi engelle."""
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text().strip())
            # Check if process is still running
            if _is_process_running(old_pid):
                print(f"[HATA] EdgeAgent zaten calisiyor (PID {old_pid})")
                print(f"       PID dosyasi: {pid_path}")
                print(f"       Durdurmak icin: taskkill /PID {old_pid} /F")
                sys.exit(1)
            else:
                # Stale PID file
                pid_path.unlink()
        except (ValueError, OSError):
            pid_path.unlink(missing_ok=True)

    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))


def remove_pid(pid_path: Path):
    """PID dosyasini sil."""
    try:
        pid_path.unlink(missing_ok=True)
    except OSError:
        pass


def _is_process_running(pid: int) -> bool:
    """Verilen PID'nin hala calisip calismadigini kontrol et."""
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, PermissionError):
        return False


# ── Mode: Inference ───────────────────────────────────────────────

def run_inference(config: dict, args):
    """
    Uretim modu: Pipeline + HMI + Watchdog + ShiftLogger + OPC-UA.
    Fabrikada 7/24 calisan ana mod.
    """
    logger = logging.getLogger("edgeagent.inference")
    shutdown_event = threading.Event()
    components = []  # (name, stop_func) tuples for graceful shutdown

    try:
        # 1. Inference Pipeline
        logger.info("=" * 60)
        logger.info("EdgeAgent Inference Pipeline baslatiliyor...")
        logger.info("=" * 60)

        from src.pipeline.inference_pipeline import InferencePipeline
        config_path = args.config
        pipeline = InferencePipeline.from_config(config_path)

        # 2. Shift Logger
        shift_logger = None
        try:
            from src.pipeline.shift_logger import ShiftLogger
            shift_cfg = config.get("shifts", {})
            shift_logger = ShiftLogger(config=shift_cfg)
            logger.info("[OK] Shift Logger baslatildi")
        except Exception as e:
            logger.warning(f"[--] Shift Logger yuklenemedi: {e}")

        # 3. Pipeline event → shift logger
        def on_event(event):
            if shift_logger:
                try:
                    verdict = event.final_verdict if hasattr(event, 'final_verdict') else "OK"
                    confidence = event.confidence if hasattr(event, 'confidence') else 1.0
                    shift_logger.record_verdict(verdict, confidence=confidence)
                except Exception:
                    pass

        pipeline.on_event(on_event)
        pipeline.start()
        components.append(("Pipeline", pipeline.stop))
        logger.info("[OK] Inference Pipeline baslatildi")

        # 4. Watchdog
        watchdog = None
        try:
            from src.pipeline.watchdog import Watchdog
            watchdog = Watchdog(pipeline)
            watchdog.start()
            components.append(("Watchdog", watchdog.stop))
            logger.info("[OK] Watchdog baslatildi")
        except Exception as e:
            logger.warning(f"[--] Watchdog yuklenemedi: {e}")

        # 5. OPC-UA Server
        opcua_cfg = config.get("opcua", {})
        if opcua_cfg.get("enabled", False):
            try:
                from src.integration.opcua_server import OPCUAServer
                opcua = OPCUAServer(config=opcua_cfg)
                # OPC-UA runs async — start in thread
                opcua_thread = threading.Thread(target=opcua.run, daemon=True)
                opcua_thread.start()
                logger.info(f"[OK] OPC-UA Server baslatildi (port {opcua_cfg.get('port', 4840)})")
            except Exception as e:
                logger.warning(f"[--] OPC-UA Server yuklenemedi: {e}")

        # 6. FastAPI HMI (ana thread'de calisir)
        hmi_cfg = config.get("hmi", {})
        host = hmi_cfg.get("host", args.host)
        port = hmi_cfg.get("port", args.port)

        logger.info("")
        logger.info(f"  +------------------------------------------+")
        logger.info(f"  |  EdgeAgent HMI: http://{host}:{port:<5}      |")
        logger.info(f"  |  Config: {Path(args.config).name:<30}|")
        logger.info(f"  |  Mode: PRODUCTION                        |")
        logger.info(f"  +------------------------------------------+")
        logger.info("")

        # Signal handler for graceful shutdown
        def _shutdown_handler(signum, frame):
            sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
            logger.info(f"Sinyal alindi: {sig_name} — kapatiliyor...")
            shutdown_event.set()

        signal.signal(signal.SIGINT, _shutdown_handler)
        signal.signal(signal.SIGTERM, _shutdown_handler)
        if sys.platform == "win32":
            try:
                signal.signal(signal.SIGBREAK, _shutdown_handler)
            except (AttributeError, OSError):
                pass

        # Wire pipeline into HMI
        from src.ui.production_hmi import app as hmi_app
        import src.ui.production_hmi as hmi_module
        hmi_module._pipeline = pipeline
        hmi_module._shift_logger = shift_logger
        hmi_module._watchdog = watchdog

        # Pipeline event → HMI recent events
        pipeline.on_event(hmi_module._on_inference_event)

        # Start uvicorn (blocking)
        import uvicorn

        uvi_config = uvicorn.Config(
            hmi_app, host=host, port=port,
            log_level="warning",  # Uvicorn kendi logunu kıs
            access_log=False,
        )
        server = uvicorn.Server(uvi_config)

        # Run server in a thread so we can catch signals
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        components.append(("HMI Server", server.shutdown))

        # Wait for shutdown signal
        shutdown_event.wait()

    except Exception as e:
        logger.error(f"Baslatma hatasi: {e}", exc_info=True)
        return 1

    finally:
        # Graceful shutdown — reverse order
        logger.info("Graceful shutdown baslatildi...")
        for name, stop_func in reversed(components):
            try:
                logger.info(f"  Durduruluyor: {name}")
                stop_func()
            except Exception as e:
                logger.error(f"  {name} durdurma hatasi: {e}")
        logger.info("Tum bilesenler durduruldu.")

    return 0


# ── Mode: Training ────────────────────────────────────────────────

def run_training(config: dict, args):
    """
    Surekli egitim modu: Retrain kosullarini kontrol et, gerekirse egit.
    Inference'dan AYRI surec olarak calisir (GPU catismasi onlenir).
    """
    logger = logging.getLogger("edgeagent.training")
    logger.info("Surekli Egitim Pipeline'i baslatildi")

    try:
        from src.mlops.continuous_trainer import ContinuousTrainer

        # Config'den CT parametrelerini al
        ct_cfg = config.get("continuous_training", {})
        trainer = ContinuousTrainer(
            min_uncertain_frames=ct_cfg.get("min_uncertain_frames", 100),
            min_feedback_corrections=ct_cfg.get("min_feedback_corrections", 50),
            min_days_between_retrain=ct_cfg.get("min_days_between_retrain", 1.0),
        )

        if args.training_action == "check":
            decision = trainer.check_retrain_needed()
            logger.info(f"Retrain karari: {decision}")
            print(json.dumps(decision.__dict__ if hasattr(decision, '__dict__') else {"result": str(decision)}, indent=2, ensure_ascii=False))
        elif args.training_action == "run":
            logger.info("Tam retrain dongusu baslatiliyor...")
            result = trainer.run_retrain_cycle()
            logger.info(f"Retrain sonucu: {result}")
        elif args.training_action == "status":
            decision = trainer.check_retrain_needed()
            status = {
                "should_retrain": decision.should_retrain,
                "reason": decision.reason,
                "uncertain_frames": decision.uncertain_count,
                "corrective_feedback": decision.feedback_corrective,
                "days_since_last_retrain": decision.days_since_last_retrain,
                "timestamp": datetime.now().isoformat(),
            }
            print(json.dumps(status, indent=2, ensure_ascii=False))
        else:
            # Varsayilan: kontrol et, gerekirse egit
            decision = trainer.check_retrain_needed()
            if decision.should_retrain:
                logger.info(f"Retrain tetiklendi: {decision.reason}")
                result = trainer.run_retrain_cycle()
                logger.info(f"Retrain tamamlandi: {result}")
            else:
                logger.info(f"Retrain gerekli degil: {decision.reason}")

    except ImportError as e:
        logger.error(f"ContinuousTrainer import hatasi: {e}")
        logger.info("Alternatif: python src/mlops/continuous_trainer.py --check")
        return 1
    except Exception as e:
        logger.error(f"Training hatasi: {e}", exc_info=True)
        return 1

    return 0


# ── Mode: Dev ─────────────────────────────────────────────────────

def run_dev(config: dict, args):
    """
    Gelistirme modu: Streamlit dashboard baslatir.
    """
    logger = logging.getLogger("edgeagent.dev")
    logger.info("Gelistirme modu — Streamlit baslatiliyor...")

    import subprocess
    dashboard = ROOT / "src" / "ui" / "sprint1_dashboard.py"

    if not dashboard.exists():
        logger.error(f"Dashboard bulunamadi: {dashboard}")
        return 1

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard),
             "--server.port", str(args.port), "--server.headless", "true"],
            cwd=str(ROOT),
        )
        return proc.returncode
    except KeyboardInterrupt:
        return 0
    except FileNotFoundError:
        logger.error("Streamlit yuklu degil: pip install streamlit")
        return 1


# ── Startup Validation ────────────────────────────────────────────

def validate_environment(mode: str) -> list:
    """Baslatma oncesi ortam kontrolu."""
    warnings = []

    # Python version
    if sys.version_info < (3, 10):
        warnings.append(f"Python 3.10+ gerekli (mevcut: {sys.version})")

    if mode in ("inference", "dev"):
        # GPU check
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                logging.getLogger("edgeagent").info(
                    f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)"
                )
            else:
                warnings.append("CUDA GPU bulunamadi — CPU modunda calisacak (yavas)")
        except ImportError:
            warnings.append("PyTorch yuklu degil")

        # Model check
        model_files = list((ROOT / "models").glob("*.pt")) + list((ROOT / "models").glob("*.onnx"))
        if not model_files:
            warnings.append("models/ klasorunde model dosyasi bulunamadi (.pt veya .onnx)")

    return warnings


# ── Banner ────────────────────────────────────────────────────────

BANNER = """
  =========================================================
  |  EdgeAgent - Endustriyel Kalite Kontrol AI Sistemi     |
  |  YOLOv10-S + CoordAtt | PaliGemma VLM | MLOps         |
  =========================================================
"""


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EdgeAgent — Unified Production Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modlar:
  inference  Pipeline + HMI + Watchdog (uretim, varsayilan)
  training   Surekli egitim kontrolu (ayri surec)
  dev        Streamlit dashboard (gelistirme)

Ornekler:
  python main.py                           # Varsayilan uretim modu
  python main.py --mode training --action check
  python main.py --mode dev --port 8501
  python main.py --config configs/custom.yaml
        """,
    )
    parser.add_argument(
        "--mode", choices=["inference", "training", "dev"],
        default="inference", help="Calisma modu (varsayilan: inference)",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(ROOT / "configs" / "production_config.yaml"),
        help="Konfigürasyon dosyasi",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HMI sunucu adresi")
    parser.add_argument("--port", type=int, default=8080, help="HMI/Streamlit portu")
    parser.add_argument("--log-level", default="INFO", help="Log seviyesi")
    parser.add_argument(
        "--training-action", choices=["check", "run", "status", "auto"],
        default="auto", help="Training modu aksiyonu",
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = ROOT / "logs"
    logger = setup_logging(log_dir, args.log_level)

    print(BANNER)
    logger.info(f"EdgeAgent v1.0 baslatiliyor | Mod: {args.mode}")
    logger.info(f"Config: {args.config}")
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Python: {sys.version.split()[0]} | Platform: {sys.platform}")

    # PID file (only for inference mode — long-running)
    pid_path = ROOT / "data" / "edgeagent.pid"
    if args.mode == "inference":
        write_pid(pid_path)
        atexit.register(remove_pid, pid_path)

    # Validate environment
    warnings = validate_environment(args.mode)
    for w in warnings:
        logger.warning(f"[UYARI] {w}")

    # Load config
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Config yuklendi: {config_path.name}")
    else:
        logger.warning(f"Config dosyasi bulunamadi: {config_path}")

    # Run selected mode
    if args.mode == "inference":
        exit_code = run_inference(config, args)
    elif args.mode == "training":
        exit_code = run_training(config, args)
    elif args.mode == "dev":
        exit_code = run_dev(config, args)
    else:
        logger.error(f"Bilinmeyen mod: {args.mode}")
        exit_code = 1

    if args.mode == "inference":
        remove_pid(pid_path)

    logger.info(f"EdgeAgent kapatildi (exit code: {exit_code})")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
