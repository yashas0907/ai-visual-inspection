$ErrorActionPreference = "Stop"
# Windows twin of build_deploy.sh: stage a lean deploy context and build.
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$OUT = "deploy\_scratch"
if (Test-Path $OUT) { Remove-Item -Recurse -Force $OUT }
New-Item -ItemType Directory -Force -Path `
  "$OUT\ml", "$OUT\backend", "$OUT\frontend", "$OUT\models\experiments\v1_resnet18" | Out-Null

# ---- app code ----
Copy-Item ml\*.py, ml\configs -Destination "$OUT\ml\" -Recurse
Copy-Item backend\app, backend\alembic, backend\alembic.ini -Destination "$OUT\backend\" -Recurse
Copy-Item frontend\src, frontend\package.json, frontend\index.html, frontend\tsconfig.json, frontend\vite.config.ts -Destination "$OUT\frontend\" -Recurse
if (Test-Path frontend\package-lock.json) { Copy-Item frontend\package-lock.json "$OUT\frontend\" }
if (Test-Path frontend\public) { Copy-Item frontend\public "$OUT\frontend\" -Recurse }

# ---- trained artifacts ----
foreach ($f in "best.pth","metadata.json","history.json","test_metrics.json","confusion_matrix.png") {
  $src = "models\experiments\v1_resnet18\$f"
  if (Test-Path $src) { Copy-Item $src "$OUT\models\experiments\v1_resnet18\" }
  else { Write-Warning "$src missing (run scripts\train.ps1 first)" }
}

# ---- deployment files ----
Copy-Item deploy\Dockerfile "$OUT\Dockerfile"
Remove-Item "$OUT\Dockerfile" -ErrorAction SilentlyContinue
Copy-Item deploy\Dockerfile "$OUT\Dockerfile"

$size = [math]::Round((Get-ChildItem $OUT -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
Write-Host "Staged $size MB in $OUT"
docker build -t surfacespec:latest $OUT
if ($LASTEXITCODE -eq 0) { Write-Host "Built surfacespec:latest" -ForegroundColor Green }
