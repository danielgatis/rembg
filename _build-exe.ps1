# Install Poetry if not already installed
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    pip install poetry
}

# Install pyinstaller
pip install pyinstaller

# Build CPU version
Write-Host "Building CPU version..." -ForegroundColor Cyan
poetry install --extras "cli cpu"
poetry run pyinstaller rembg.spec
Rename-Item -Path "dist/rembg" -NewName "rembg-cpu"

# Build GPU version
Write-Host "Building GPU version..." -ForegroundColor Cyan
poetry install --extras "cli gpu"
poetry run pyinstaller rembg.spec --noconfirm
Rename-Item -Path "dist/rembg" -NewName "rembg-gpu"

Write-Host "Build complete!" -ForegroundColor Green
Write-Host "CPU version: dist/rembg-cpu"
Write-Host "GPU version: dist/rembg-gpu"
