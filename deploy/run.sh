#!/usr/bin/env bash
# EdgeAgent — Linux/Jetson Manuel Baslatici
# Servis olarak degil, terminal uzerinden calistirmak icin.
# Servis kurulumu icin: deploy/setup-jetson.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo ""
echo "  EdgeAgent Baslatiliyor..."
echo "  Calisma dizini: $PROJECT_DIR"
echo ""

# venv varsa aktive et
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "  [OK] Virtual environment aktif"
else
    echo "  [--] Virtual environment bulunamadi (.venv)"
    echo "       Olusturmak icin: python3 -m venv .venv"
fi

# CUDA kontrolu
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  [OK] CUDA: {torch.cuda.get_device_name(0)}')
else:
    print('  [--] CUDA yok — CPU modunda calisacak')
" 2>/dev/null || {
    echo "  [HATA] Python veya PyTorch bulunamadi"
    echo "         pip install -r requirements.txt"
    exit 1
}

echo ""
echo "  Durdurmak icin: Ctrl+C"
echo ""

# Ana uygulamayi baslat
exec python3 main.py "$@"
