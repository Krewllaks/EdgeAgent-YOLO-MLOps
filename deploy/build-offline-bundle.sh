#!/usr/bin/env bash
# EdgeAgent — Offline Bundle Builder
# Internetsiz (air-gapped) fabrikalara kurulum icin.
#
# INTERNET OLAN bir makinede calistirin:
#   bash deploy/build-offline-bundle.sh
#
# Cikti: edgeagent-offline-bundle/ klasoru (~4GB)
# Bu klasoru USB ile fabrika PC'sine kopyalayin.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUNDLE_DIR="$PROJECT_DIR/edgeagent-offline-bundle"

echo ""
echo "  EdgeAgent Offline Bundle Olusturuluyor..."
echo "  Cikti: $BUNDLE_DIR"
echo ""

# Temizle
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/deploy/wheels"

# ── 1. Pip wheel'leri indir ─────────────────────────────────
echo "[1/4] Python paketleri indiriliyor (bu uzun surebilir)..."
pip download \
    -d "$BUNDLE_DIR/deploy/wheels/" \
    -r "$PROJECT_DIR/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "  [OK] $(ls "$BUNDLE_DIR/deploy/wheels/" | wc -l) wheel indirildi"

# ── 2. Kaynak kodunu kopyala ────────────────────────────────
echo "[2/4] Kaynak kodu kopyalaniyor..."
cp "$PROJECT_DIR/main.py" "$BUNDLE_DIR/"
cp "$PROJECT_DIR/requirements.txt" "$BUNDLE_DIR/"
cp -r "$PROJECT_DIR/src" "$BUNDLE_DIR/"
cp -r "$PROJECT_DIR/configs" "$BUNDLE_DIR/"
cp -r "$PROJECT_DIR/scripts" "$BUNDLE_DIR/"
cp -r "$PROJECT_DIR/deploy" "$BUNDLE_DIR/" 2>/dev/null || true

# wheels zaten kopyalandi, deploy icindekini koru
echo "  [OK] Kaynak kodu kopyalandi"

# ── 3. Model dosyalari ─────────────────────────────────────
echo "[3/4] Model dosyalari kopyalaniyor..."
mkdir -p "$BUNDLE_DIR/models"
if ls "$PROJECT_DIR"/models/*.pt 1>/dev/null 2>&1; then
    cp "$PROJECT_DIR"/models/*.pt "$BUNDLE_DIR/models/"
fi
if ls "$PROJECT_DIR"/models/*.onnx 1>/dev/null 2>&1; then
    cp "$PROJECT_DIR"/models/*.onnx "$BUNDLE_DIR/models/"
fi
echo "  [OK] Modeller kopyalandi"

# ── 4. Offline installer scriptleri ─────────────────────────
echo "[4/4] Installer scriptleri olusturuluyor..."

# install-offline.bat (Windows)
cat > "$BUNDLE_DIR/install-offline.bat" << 'BATEOF'
@echo off
echo EdgeAgent Offline Kurulum
echo.

cd /d "%~dp0"

:: venv olustur
python -m venv .venv
if errorlevel 1 (
    echo [HATA] Python bulunamadi. Python 3.10+ yukleyin.
    pause
    exit /b 1
)

:: Paketleri yukle (offline)
.venv\Scripts\pip install --no-index --find-links=deploy\wheels\ -r requirements.txt
if errorlevel 1 (
    echo [HATA] Paket yuklemesi basarisiz
    pause
    exit /b 1
)

:: Config kopyala
if not exist configs\production_config.local.yaml (
    copy configs\production_config.yaml configs\production_config.local.yaml
    echo [OK] Konfigurasyon dosyasi olusturuldu: configs\production_config.local.yaml
    echo      Bu dosyayi fabrika ayarlarina gore duzenleyin.
)

echo.
echo [OK] Kurulum tamamlandi!
echo.
echo Sonraki adimlar:
echo   1. configs\production_config.local.yaml dosyasini duzenleyin
echo   2. deploy\run.bat ile calistirin
echo   3. Veya deploy\install-service.ps1 ile servis olarak kurun
echo.
pause
BATEOF

# install-offline.sh (Linux)
cat > "$BUNDLE_DIR/install-offline.sh" << 'SHEOF'
#!/usr/bin/env bash
set -e
echo "EdgeAgent Offline Kurulum"
echo ""

cd "$(dirname "$0")"

# venv olustur
python3 -m venv .venv

# Paketleri yukle (offline)
.venv/bin/pip install --no-index --find-links=deploy/wheels/ -r requirements.txt

# Config kopyala
if [ ! -f configs/production_config.local.yaml ]; then
    cp configs/production_config.yaml configs/production_config.local.yaml
    echo "[OK] Konfigurasyon: configs/production_config.local.yaml"
    echo "     Bu dosyayi fabrika ayarlarina gore duzenleyin."
fi

echo ""
echo "[OK] Kurulum tamamlandi!"
echo ""
echo "Sonraki adimlar:"
echo "  1. configs/production_config.local.yaml dosyasini duzenleyin"
echo "  2. bash deploy/run.sh ile calistirin"
echo "  3. Veya sudo bash deploy/setup-jetson.sh ile servis kurun"
SHEOF

chmod +x "$BUNDLE_DIR/install-offline.sh"
chmod +x "$BUNDLE_DIR/deploy/run.sh" 2>/dev/null || true

echo "  [OK] Installer scriptleri olusturuldu"

# Boyut raporu
BUNDLE_SIZE=$(du -sh "$BUNDLE_DIR" | awk '{print $1}')
echo ""
echo "  ══════════════════════════════════════════"
echo "  Bundle hazir: $BUNDLE_DIR"
echo "  Boyut: $BUNDLE_SIZE"
echo "  ══════════════════════════════════════════"
echo ""
echo "  USB ile fabrikaya kopyalayin, sonra:"
echo "    Windows: install-offline.bat"
echo "    Linux:   bash install-offline.sh"
echo ""
