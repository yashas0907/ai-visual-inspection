# Free live demo: run the platform locally and expose it on a public URL.
#
# Uses a Cloudflare quick tunnel (cloudflared) — no account, no token, no
# card. Anyone can open the printed https://*.trycloudflare.com link and use
# the app while your laptop runs it.
#
# Usage:  scripts\share_demo.ps1
# Stop:   Ctrl+C in this window (kills the tunnel), then it also stops the app.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

# ---- 0. Preflight checks ----------------------------------------------------
if (-not (Test-Path "models\experiments\v1_resnet18\best.pth")) {
    Write-Host "Model missing. Run scripts\train.ps1 first." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path "frontend\dist\index.html")) {
    Write-Host "Frontend build missing. Building..." -ForegroundColor Yellow
    Push-Location frontend; npm run build; Pop-Location
}

# ---- 1. Start the app (single-origin SPA mode) ------------------------------
$port = 7860
Write-Host "[1/3] starting SurfaceSpec on http://localhost:$port ..." -ForegroundColor Cyan
$env:PYTHONPATH = "$root;$root\backend"
$env:VI_FRONTEND_DIST = "$root\frontend\dist"
$app = Start-Process -FilePath "powershell" -ArgumentList "-NoProfile", "-Command", `
    "Set-Location '$root'; `$env:PYTHONPATH='$root;$root\backend'; `$env:VI_FRONTEND_DIST='$root\frontend\dist'; & '$root\.venv\Scripts\python.exe' -m uvicorn app.main:app --app-dir backend --port $port *> share_demo_app.log" `
    -WindowStyle Hidden -PassThru

# Retry loop: model load takes ~15-25s, don't fail on the first probe.
$ready = $false
for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep 5
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:$port/api/health" -TimeoutSec 5
        if ($health.status -eq "ok") { $ready = $true; break }
    } catch { }
}
if (-not $ready) {
    Write-Host "App failed to start — check share_demo_app.log" -ForegroundColor Red
    Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "      app ready (model $($health.model_version))." -ForegroundColor Green

# ---- 2. Get cloudflared (single binary, no install) --------------------------
$tools = Join-Path $root ".tools"
$cf = Join-Path $tools "cloudflared.exe"
if (-not (Test-Path $cf)) {
    Write-Host "[2/3] downloading cloudflared (~50MB, one time) ..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $tools | Out-Null
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest `
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" `
        -OutFile $cf -UseBasicParsing
} else {
    Write-Host "[2/3] cloudflared found." -ForegroundColor Cyan
}

# ---- 3. Start the tunnel and read the public URL ------------------------------
Write-Host "[3/3] opening public tunnel (Ctrl+C here to stop sharing) ..." -ForegroundColor Cyan
$log = Join-Path $root "share_demo_tunnel.log"
Remove-Item $log -ErrorAction SilentlyContinue
$cfProc = Start-Process -FilePath $cf `
    -ArgumentList "tunnel", "--url", "http://127.0.0.1:$port", "--no-autoupdate" `
    -RedirectStandardError $log -PassThru -WindowStyle Hidden

# cloudflared prints the trycloudflare URL to stderr within a few seconds
$url = $null
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep 1
    if (Test-Path $log) {
        $m = Select-String -Path $log -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue |
             Select-Object -First 1
        if ($m) { $url = $m.Matches[0].Value; break }
    }
    if ($cfProc.HasExited) { break }
}

if ($url) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host " LIVE DEMO URL (share this in interviews / with anyone):" -ForegroundColor Green
    Write-Host "   $url" -ForegroundColor White
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host " Local copy:   http://localhost:$port"
    Write-Host " Keep this window OPEN while demoing."
    Write-Host " Stop sharing: press Ctrl+C, then Enter."
    try {
        # Keep this PowerShell alive until Ctrl+C; then clean up.
        while (-not $cfProc.HasExited) { Start-Sleep 2 }
    } finally {
        Stop-Process -Id $cfProc.Id -Force -ErrorAction SilentlyContinue
        Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue
        Write-Host "Demo stopped. App and tunnel closed." -ForegroundColor Yellow
    }
} else {
    Write-Host "Tunnel failed to produce a URL — check share_demo_tunnel.log" -ForegroundColor Red
    Get-Content $log -Tail 15 -ErrorAction SilentlyContinue
    Stop-Process -Id $cfProc.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue
}
