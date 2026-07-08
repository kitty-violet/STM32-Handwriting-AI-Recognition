$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Write-Host "检查目录: $Root"

$Required = @(
  "01_STM32固件源码\Project\RVMDK（uv5）\BH-F103.uvprojx",
  "01_STM32固件源码\User\wifi_report.h",
  "01_STM32固件源码\prebuilt\Template_demo_placeholder_wifi.hex",
  "02_PC端网页与训练工具\网页接收端\web_receiver.py",
  "02_PC端网页与训练工具\requirements.txt",
  "03_TF卡示例文件\MODEL\FNN.BIN",
  "04_项目说明文档\使用说明.md",
  "README.md"
)
$Missing = @()
foreach ($rel in $Required) {
  if (!(Test-Path -LiteralPath (Join-Path $Root $rel))) { $Missing += $rel }
}
if ($Missing.Count -gt 0) {
  Write-Host "缺少关键文件:" -ForegroundColor Red
  $Missing | ForEach-Object { Write-Host "  $_" }
  exit 1
}

$Sensitive = "Kuang|3044234544|Tencent|QQ|Redmi K7|192\.168\.13\.173|192\.168\.13\.168|kt12345678|E:\\|G:\\|ANACONODA|Automation 2408|D:\\QQ|v1\.1基础|v8\.0|deepseek_chrome_profile|Lenovo"
$Self = $MyInvocation.MyCommand.Path
$Hits = & rg -n --hidden --glob '!.git/**' --glob '!05_辅助工具/检查开源发布包.ps1' --glob '!*.pth' --glob '!*.bin' --glob '!*.hex' --glob '!*.BMP' --glob '!*.bmp' --glob '!*.FON' --glob '!*.xlsx' $Sensitive $Root 2>$null
if ($LASTEXITCODE -eq 0 -and $Hits.Count -gt 0) {
  Write-Host "发现可能的个人信息残留:" -ForegroundColor Red
  $Hits
  exit 1
}

$AllowedHex = Join-Path $Root "01_STM32固件源码\prebuilt\Template_demo_placeholder_wifi.hex"
$Bad = Get-ChildItem -LiteralPath $Root -Recurse -Force -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -notlike (Join-Path $Root ".git") + "*" } |
  Where-Object {
    $_.Name -in @("__pycache__", "deepseek_chrome_profile", "本机浏览器登录缓存", ".idea", "Output", "Listing", "Objects") -or
    ($_.Extension -ieq ".hex" -and $_.FullName -ne $AllowedHex) -or
    $_.Extension -in @(".pyc", ".pyo", ".log", ".bak", ".axf", ".map", ".lst", ".o", ".d", ".crf", ".lnp") -or
    $_.Name -like "*.uvguix.*"
  }
if ($Bad.Count -gt 0) {
  Write-Host "发现不建议发布的生成物:" -ForegroundColor Red
  $Bad | Select-Object -First 80 -ExpandProperty FullName
  exit 1
}

$HexText = [System.IO.File]::ReadAllText($AllowedHex)
$HexSensitive = "Redmi|kt123|192\.168\.13|Kuang|Automation 2408|Lenovo|Tencent|3044234544|ANACONODA"
if ($HexText -match $HexSensitive) {
  Write-Host "示例 HEX 中疑似包含个人信息，请重新使用占位 WiFi 配置编译。" -ForegroundColor Red
  exit 1
}

$Bytes = New-Object System.Collections.Generic.List[byte]
foreach ($Line in [System.IO.File]::ReadLines($AllowedHex)) {
  if (!$Line.StartsWith(":") -or $Line.Length -lt 11) { continue }
  $Count = [Convert]::ToInt32($Line.Substring(1, 2), 16)
  $Type = [Convert]::ToInt32($Line.Substring(7, 2), 16)
  if ($Type -ne 0) { continue }
  for ($i = 0; $i -lt $Count; $i++) {
    $Bytes.Add([Convert]::ToByte($Line.Substring(9 + $i * 2, 2), 16))
  }
}
$HexAscii = [System.Text.Encoding]::ASCII.GetString($Bytes.ToArray())
if ($HexAscii -match $HexSensitive) {
  Write-Host "示例 HEX 解码后疑似包含个人信息，请重新使用占位 WiFi 配置编译。" -ForegroundColor Red
  exit 1
}

Write-Host "检查通过：关键文件、脱敏、发布清洁度均正常。" -ForegroundColor Green
