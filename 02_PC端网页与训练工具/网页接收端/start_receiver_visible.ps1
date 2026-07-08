$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$ip = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike "169.254.*" -and $_.IPAddress -ne "127.0.0.1" } |
    Select-Object -First 1 -ExpandProperty IPAddress)
if (-not $ip) { $ip = "YOUR_PC_IP" }

Write-Host "=== STM32 web_receiver visible console ==="
Write-Host "Using Python from PATH"
Write-Host "PC server IP should be: $ip:8000"
Write-Host "Web page: http://127.0.0.1:8080/"
Write-Host "Only keep this one receiver window open during demo."
Write-Host ""

python -u ".\web_receiver.py" --deepseek
Read-Host "Press Enter to exit"

