"""
MLflow Server Launcher — Kalici ve uzaktan erisimli MLflow.

Sorun: mlflow.db SQLite dosyasi yerel, her seferinde sifirlanabiliyor
ve baska PC'den erisilemez.

Cozum: MLflow'u server modunda calistir:
- Veriler kalici olarak saklanir (SQLite + artifacts)
- Baska PC'lerden http://<senin-ip>:5000 ile erisilebilir
- Dashboard UI tarayicidan acilabilir

Kullanim:
    python scripts/start_mlflow_server.py              # Varsayilan (localhost:5000)
    python scripts/start_mlflow_server.py --host 0.0.0.0  # Tum agdan erisim
    python scripts/start_mlflow_server.py --port 5050   # Farkli port
"""

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get_local_ip() -> str:
    """Get the local network IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    parser = argparse.ArgumentParser(description="MLflow Server Launcher")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind (0.0.0.0 = tum agdan erisim, 127.0.0.1 = sadece yerel)",
    )
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--backend-store", type=str,
        default=str(ROOT / "mlflow.db"),
        help="MLflow backend store (SQLite DB path)",
    )
    parser.add_argument(
        "--artifact-root", type=str,
        default=str(ROOT / "runs" / "mlflow"),
        help="MLflow artifact storage directory",
    )
    args = parser.parse_args()

    # Ensure directories exist
    Path(args.artifact_root).mkdir(parents=True, exist_ok=True)
    Path(args.backend_store).parent.mkdir(parents=True, exist_ok=True)

    local_ip = get_local_ip()

    print("=" * 60)
    print("  MLflow Server")
    print("=" * 60)
    print(f"  Backend DB : {args.backend_store}")
    print(f"  Artifacts  : {args.artifact_root}")
    print(f"  Host       : {args.host}")
    print(f"  Port       : {args.port}")
    print()
    print(f"  Yerel erisim : http://localhost:{args.port}")
    print(f"  Ag erisimi   : http://{local_ip}:{args.port}")
    print()
    print("  Arkadasin bu URL'yi tarayicida acarak deneyleri gorebilir.")
    print("  Durdurmak icin Ctrl+C basin.")
    print("=" * 60)
    print()

    # Build mlflow server command
    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", f"sqlite:///{args.backend_store}",
        "--default-artifact-root", args.artifact_root,
        "--host", args.host,
        "--port", str(args.port),
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[OK] MLflow server durduruldu.")


if __name__ == "__main__":
    main()
