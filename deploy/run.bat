@echo off
:: EdgeAgent — Windows Manuel Baslatici
:: Servis olarak degil, terminal uzerinden calistirmak icin.
:: Servis kurulumu icin: deploy\install-service.ps1

cd /d "%~dp0\.."
echo.
echo   EdgeAgent Baslatiliyor...
echo   Calisma dizini: %CD%
echo.

:: venv varsa aktive et
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo   [OK] Virtual environment aktif
) else (
    echo   [--] Virtual environment bulunamadi (.venv)
    echo       Olusturmak icin: python -m venv .venv
)

:: CUDA kontrolu
python -c "import torch; print('   [OK] CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK (CPU modu)')" 2>nul
if errorlevel 1 (
    echo   [HATA] Python veya PyTorch bulunamadi
    echo          pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo   Durdurmak icin: Ctrl+C
echo.

:: Ana uygulamayi baslat
python main.py %*
if errorlevel 1 (
    echo.
    echo   [HATA] EdgeAgent beklenmedik sekilde kapandi (kod: %ERRORLEVEL%)
    pause
)
