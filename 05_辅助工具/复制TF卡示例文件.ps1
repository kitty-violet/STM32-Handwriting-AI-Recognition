param(
  [Parameter(Mandatory=$true)]
  [string]$TfRoot
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Source = Join-Path $Root "03_TF卡示例文件"
if (!(Test-Path -LiteralPath $Source)) { throw "找不到 TF 卡示例目录: $Source" }
if (!(Test-Path -LiteralPath $TfRoot)) { throw "找不到 TF 卡盘符/目录: $TfRoot" }

Write-Host "将复制 $Source 到 $TfRoot"
Copy-Item -LiteralPath (Join-Path $Source "*") -Destination $TfRoot -Recurse -Force
Write-Host "复制完成。请确认 TF 卡根目录包含 MODEL、TEST、Font、WEB 等目录。" -ForegroundColor Green
