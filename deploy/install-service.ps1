# EdgeAgent — Windows Service Installer (NSSM)
# Yonetici olarak calistirin: powershell -ExecutionPolicy Bypass -File deploy\install-service.ps1
#
# Bu script:
# 1. NSSM (Non-Sucking Service Manager) indirir
# 2. "EdgeAgent" Windows servisi olusturur
# 3. Otomatik baslatma + crash recovery ayarlar
# 4. Opsiyonel: "EdgeAgentTrainer" zamanli gorev olusturur
#
# Kaldirmak icin: deploy\uninstall-service.ps1

param(
    [string]$ProjectDir = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$ServiceName = "EdgeAgent",
    [string]$TrainerTaskName = "EdgeAgentTrainer",
    [int]$HmiPort = 8080,
    [switch]$SkipTrainer,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

# ── Yonetici kontrolu ────────────────────────────────────────
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "[HATA] Bu scripti Yonetici olarak calistirin!" -ForegroundColor Red
    exit 1
}

# ── NSSM path ────────────────────────────────────────────────
$NssmDir = Join-Path $PSScriptRoot "nssm"
$NssmExe = Join-Path $NssmDir "nssm.exe"

if (-not (Test-Path $NssmExe)) {
    Write-Host "[INFO] NSSM indiriliyor..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $NssmDir | Out-Null

    $nssmUrl = "https://nssm.cc/release/nssm-2.24.zip"
    $zipPath = Join-Path $env:TEMP "nssm.zip"
    $extractPath = Join-Path $env:TEMP "nssm_extract"

    try {
        Invoke-WebRequest -Uri $nssmUrl -OutFile $zipPath -UseBasicParsing
        Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
        $nssmBin = Get-ChildItem -Path $extractPath -Recurse -Filter "nssm.exe" |
            Where-Object { $_.DirectoryName -like "*win64*" } |
            Select-Object -First 1
        Copy-Item $nssmBin.FullName $NssmExe
        Remove-Item $zipPath -Force
        Remove-Item $extractPath -Recurse -Force
        Write-Host "[OK] NSSM indirildi: $NssmExe" -ForegroundColor Green
    }
    catch {
        Write-Host "[HATA] NSSM indirilemedi. Manuel indirin: https://nssm.cc" -ForegroundColor Red
        Write-Host "       $NssmExe konumuna kopyalayin." -ForegroundColor Yellow
        exit 1
    }
}

# ── Python path ───────────────────────────────────────────────
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $VenvPython) {
        Write-Host "[HATA] Python bulunamadi. .venv olusturun: python -m venv .venv" -ForegroundColor Red
        exit 1
    }
}

$MainPy = Join-Path $ProjectDir "main.py"
if (-not (Test-Path $MainPy)) {
    Write-Host "[HATA] main.py bulunamadi: $MainPy" -ForegroundColor Red
    exit 1
}

# ── Log dizini ────────────────────────────────────────────────
$LogDir = Join-Path $ProjectDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host ""
Write-Host "  ╔═══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║  EdgeAgent Windows Service Installer      ║" -ForegroundColor Cyan
Write-Host "  ╚═══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Proje:    $ProjectDir"
Write-Host "  Python:   $VenvPython"
Write-Host "  Servis:   $ServiceName"
Write-Host "  HMI Port: $HmiPort"
Write-Host ""

# ── Inference Servisi Kur ─────────────────────────────────────
Write-Host "[1/3] $ServiceName servisi olusturuluyor..." -ForegroundColor Cyan

# Mevcut servisi kaldir
& $NssmExe stop $ServiceName 2>$null
& $NssmExe remove $ServiceName confirm 2>$null

# Yeni servis olustur
& $NssmExe install $ServiceName $VenvPython
& $NssmExe set $ServiceName AppParameters "main.py --config configs/production_config.yaml --port $HmiPort"
& $NssmExe set $ServiceName AppDirectory $ProjectDir
& $NssmExe set $ServiceName DisplayName "EdgeAgent Kalite Kontrol"
& $NssmExe set $ServiceName Description "YOLOv10-S + CoordAtt endüstriyel kalite kontrol sistemi"
& $NssmExe set $ServiceName Start SERVICE_AUTO_START
& $NssmExe set $ServiceName ObjectName LocalSystem

# Logging
& $NssmExe set $ServiceName AppStdout (Join-Path $LogDir "service_stdout.log")
& $NssmExe set $ServiceName AppStderr (Join-Path $LogDir "service_stderr.log")
& $NssmExe set $ServiceName AppStdoutCreationDisposition 4  # Append
& $NssmExe set $ServiceName AppStderrCreationDisposition 4  # Append
& $NssmExe set $ServiceName AppRotateFiles 1
& $NssmExe set $ServiceName AppRotateBytes 10485760  # 10MB

# Crash recovery: restart after 5s, 10s, 30s
& $NssmExe set $ServiceName AppRestartDelay 5000
sc.exe failure $ServiceName reset= 86400 actions= restart/5000/restart/10000/restart/30000

Write-Host "[OK] $ServiceName servisi olusturuldu" -ForegroundColor Green

# ── Trainer Zamanli Gorev ─────────────────────────────────────
if (-not $SkipTrainer) {
    Write-Host "[2/3] $TrainerTaskName zamanli gorevi olusturuluyor..." -ForegroundColor Cyan

    # Mevcut gorevi kaldir
    Unregister-ScheduledTask -TaskName $TrainerTaskName -Confirm:$false -ErrorAction SilentlyContinue

    $Action = New-ScheduledTaskAction `
        -Execute $VenvPython `
        -Argument "main.py --mode training --training-action auto" `
        -WorkingDirectory $ProjectDir

    # Her 6 saatte bir calistir
    $Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 6)

    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2)

    Register-ScheduledTask `
        -TaskName $TrainerTaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Description "EdgeAgent surekli egitim kontrolu (6 saatte bir)" `
        -RunLevel Highest

    Write-Host "[OK] $TrainerTaskName zamanli gorevi olusturuldu (6 saatte bir)" -ForegroundColor Green
} else {
    Write-Host "[2/3] Trainer atlandi (-SkipTrainer)" -ForegroundColor Yellow
}

# ── Servisi Baslat ────────────────────────────────────────────
Write-Host "[3/3] Servis baslatiliyor..." -ForegroundColor Cyan
& $NssmExe start $ServiceName
Start-Sleep -Seconds 3

$status = (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue).Status
if ($status -eq "Running") {
    Write-Host ""
    Write-Host "  ✓ EdgeAgent servisi calisiyor!" -ForegroundColor Green
    Write-Host "  ✓ HMI: http://localhost:$HmiPort" -ForegroundColor Green
    Write-Host "  ✓ Loglar: $LogDir" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Yonetim komutlari:" -ForegroundColor Cyan
    Write-Host "    nssm stop $ServiceName        # Durdur"
    Write-Host "    nssm start $ServiceName       # Baslat"
    Write-Host "    nssm restart $ServiceName     # Yeniden baslat"
    Write-Host "    nssm status $ServiceName      # Durum"
    Write-Host ""
} else {
    Write-Host "[UYARI] Servis baslatildi ama durumu: $status" -ForegroundColor Yellow
    Write-Host "        Loglari kontrol edin: $LogDir\service_stderr.log"
}
