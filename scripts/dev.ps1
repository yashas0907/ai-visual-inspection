#!/usr/bin/env pwsh
<#
Start the full local dev stack:
- backend  : uvicorn on :8000
- frontend: vite dev server on :5173 (proxies /api)
#>
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$env:PYTHONPATH = "$root;$root\backend"

Write-Host "[dev] starting backend on :8000" -ForegroundColor Cyan
Start-Process -FilePath "powershell" -ArgumentList "-NoProfile", "-Command", `
    "Set-Location '$root'; `$env:PYTHONPATH='$root;$root\backend'; & '$root\.venv\Scripts\python.exe' -m uvicorn app.main:app --app-dir backend --reload --port 8000 2>&1 | Out-File -Encoding utf8 backend_dev.log"

Write-Host "[dev] starting frontend on :5173" -ForegroundColor Cyan
Start-Process -FilePath "powershell" -ArgumentList "-NoProfile", "-Command", `
    "Set-Location '$root\frontend'; npm run dev 2>&1 | Out-File -Encoding utf8 frontend_dev.log"

Write-Host "Backend:  http://localhost:8000/api/docs" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Green
