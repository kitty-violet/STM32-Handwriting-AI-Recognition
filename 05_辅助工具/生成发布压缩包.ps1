$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Parent = Split-Path -Parent $Root
$Name = Split-Path -Leaf $Root
$Zip = Join-Path $Parent ("$Name.zip")
if (Test-Path -LiteralPath $Zip) { Remove-Item -LiteralPath $Zip -Force }
Compress-Archive -LiteralPath $Root -DestinationPath $Zip -Force
Write-Host "已生成: $Zip" -ForegroundColor Green
