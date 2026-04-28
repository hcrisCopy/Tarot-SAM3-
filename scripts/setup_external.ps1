$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path external | Out-Null

if (-not (Test-Path "external/sam3/.git")) {
  git clone https://github.com/facebookresearch/sam3.git external/sam3
}

if (-not (Test-Path "external/refer/.git")) {
  git clone https://github.com/lichengunc/refer.git external/refer
}

Write-Host "External repositories are ready."
Write-Host "Next:"
Write-Host "  cd external/sam3"
Write-Host "  pip install -e ."
Write-Host "  pip install -e `".[notebooks]`""

