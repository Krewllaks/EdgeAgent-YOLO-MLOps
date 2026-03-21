#!/usr/bin/env bash
# EdgeAgent — Jetson Orin Nano One-Click Setup
# Kullanim: sudo bash deploy/setup-jetson.sh
#
# Bu script:
# 1. edgeagent kullanicisi olusturur
# 2. Sistem bagimliliklerini yukler
# 3. Python venv + pip paketleri kurar
# 4. systemd servislerini kaydeder
# 5. Servisleri baslatir

set -e

INSTALL_DIR="/opt/edgeagent"
USER_NAME="edgeagent"

echo ""
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║  EdgeAgent Jetson Setup                   ║"
echo "  ╚═══════════════════════════════════════════╝"
echo ""

# Root kontrolu
if [ "$EUID" -ne 0 ]; then
    echo "[HATA] Root olarak calistirin: sudo bash $0"
    exit 1
fi

# ── 1. Kullanici olustur ────────────────────────────────────
echo "[1/6] Kullanici olusturuluyor..."
if ! id "$USER_NAME" &>/dev/null; then
    useradd -r -s /bin/false -d "$INSTALL_DIR" "$USER_NAME"
    usermod -aG video "$USER_NAME"  # GPU erisimiroot
    echo "  [OK] $USER_NAME kullanicisi olusturuldu"
else
    echo "  [--] $USER_NAME zaten var"
fi

# ── 2. Sistem bagimliliklari ────────────────────────────────
echo "[2/6] Sistem bagimliliklari yukleniyor..."
apt-get update -qq
apt-get install -y -qq \
    python3 python3-venv python3-pip \
    libopencv-dev \
    mosquitto mosquitto-clients \
    curl

# Mosquitto ayarla
systemctl enable mosquitto
systemctl start mosquitto
echo "  [OK] Sistem bagimliliklari yuklendi"

# ── 3. Proje dosyalari ─────────────────────────────────────
echo "[3/6] Proje dosyalari kopyalaniyor..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ "$PROJECT_DIR" != "$INSTALL_DIR" ]; then
    mkdir -p "$INSTALL_DIR"
    cp -r "$PROJECT_DIR/main.py" "$INSTALL_DIR/"
    cp -r "$PROJECT_DIR/src" "$INSTALL_DIR/"
    cp -r "$PROJECT_DIR/configs" "$INSTALL_DIR/"
    cp -r "$PROJECT_DIR/scripts" "$INSTALL_DIR/"
    cp -r "$PROJECT_DIR/requirements.txt" "$INSTALL_DIR/"
    mkdir -p "$INSTALL_DIR/models" "$INSTALL_DIR/data" "$INSTALL_DIR/logs"

    # Model dosyalarini kopyala
    if ls "$PROJECT_DIR"/models/*.pt 1>/dev/null 2>&1; then
        cp "$PROJECT_DIR"/models/*.pt "$INSTALL_DIR/models/"
    fi
    if ls "$PROJECT_DIR"/models/*.onnx 1>/dev/null 2>&1; then
        cp "$PROJECT_DIR"/models/*.onnx "$INSTALL_DIR/models/"
    fi
fi

chown -R "$USER_NAME:$USER_NAME" "$INSTALL_DIR"
echo "  [OK] Dosyalar kopyalandi: $INSTALL_DIR"

# ── 4. Python venv + bagimlilklar ──────────────────────────
echo "[4/6] Python ortami hazirlaniyor..."
cd "$INSTALL_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Jetson icin PyTorch: NVIDIA'nin wheel repo'sundan yukle
JETPACK_VERSION=$(dpkg -l | grep nvidia-jetpack | awk '{print $3}' | head -1 || echo "unknown")
echo "  JetPack: $JETPACK_VERSION"

# Offline wheel'ler varsa kullan
if [ -d "deploy/wheels" ] && [ "$(ls -A deploy/wheels 2>/dev/null)" ]; then
    echo "  Offline wheels kullaniliyor..."
    .venv/bin/pip install --no-index --find-links=deploy/wheels/ -r requirements.txt
else
    echo "  Online kurulum..."
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

chown -R "$USER_NAME:$USER_NAME" "$INSTALL_DIR"
echo "  [OK] Python bagimliliklari yuklendi"

# ── 5. systemd servisleri ──────────────────────────────────
echo "[5/6] systemd servisleri kuruluyor..."
cp "$INSTALL_DIR/deploy/edgeagent.service" /etc/systemd/system/
cp "$INSTALL_DIR/deploy/edgeagent-trainer.service" /etc/systemd/system/
cp "$INSTALL_DIR/deploy/edgeagent-trainer.timer" /etc/systemd/system/

systemctl daemon-reload
systemctl enable edgeagent
systemctl enable edgeagent-trainer.timer
echo "  [OK] Servisler kaydedildi"

# ── 6. Servisleri baslat ───────────────────────────────────
echo "[6/6] Servisler baslatiliyor..."
systemctl start edgeagent
systemctl start edgeagent-trainer.timer

sleep 3

STATUS=$(systemctl is-active edgeagent)
echo ""
if [ "$STATUS" = "active" ]; then
    echo "  ✓ EdgeAgent calisiyor!"
    echo "  ✓ HMI: http://$(hostname -I | awk '{print $1}'):8080"
    echo "  ✓ Loglar: journalctl -u edgeagent -f"
    echo ""
    echo "  Yonetim komutlari:"
    echo "    systemctl status edgeagent        # Durum"
    echo "    systemctl restart edgeagent       # Yeniden baslat"
    echo "    systemctl stop edgeagent          # Durdur"
    echo "    journalctl -u edgeagent -f        # Canli log"
    echo "    systemctl list-timers             # Timer durumu"
else
    echo "  [UYARI] Servis durumu: $STATUS"
    echo "  Log kontrol: journalctl -u edgeagent -n 50"
fi
echo ""
