# EdgeAgent — Windows Service Uninstaller
# Yonetici olarak calistirin.

param(
    [string]$ServiceName = "EdgeAgent",
    [string]$TrainerTaskName = "EdgeAgentTrainer"
)

$NssmExe = Join-Path $PSScriptRoot "nssm\nssm.exe"

Write-Host ""
Write-Host "  EdgeAgent servisleri kaldiriliyor..." -ForegroundColor Cyan
Write-Host ""

# Stop and remove service
if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
    & $NssmExe stop $ServiceName 2>$null
    Start-Sleep -Seconds 2
    & $NssmExe remove $ServiceName confirm
    Write-Host "  [OK] $ServiceName servisi kaldirildi" -ForegroundColor Green
} else {
    Write-Host "  [--] $ServiceName servisi bulunamadi" -ForegroundColor Yellow
}

# Remove scheduled task
if (Get-ScheduledTask -TaskName $TrainerTaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TrainerTaskName -Confirm:$false
    Write-Host "  [OK] $TrainerTaskName zamanli gorevi kaldirildi" -ForegroundColor Green
} else {
    Write-Host "  [--] $TrainerTaskName zamanli gorevi bulunamadi" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  Tamamlandi. Log ve veri dosyalari korundu." -ForegroundColor Cyan
Write-Host ""
