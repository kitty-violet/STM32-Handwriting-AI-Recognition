@echo off
chcp 65001 >nul
set "KEIL=%KEIL_UV4%"
if "%KEIL%"=="" if exist "C:\Keil_v5\UV4\UV4.exe" set "KEIL=C:\Keil_v5\UV4\UV4.exe"
if "%KEIL%"=="" if exist "C:\Keil\UV4\UV4.exe" set "KEIL=C:\Keil\UV4\UV4.exe"
if "%KEIL%"=="" if exist "%ProgramFiles%\Keil_v5\UV4\UV4.exe" set "KEIL=%ProgramFiles%\Keil_v5\UV4\UV4.exe"
set "PROJ="
for /d %%D in ("%~dp0Project\RVMDK*") do (
    if exist "%%~fD\BH-F103.uvprojx" set "PROJ=%%~fD\BH-F103.uvprojx"
)
if not exist "%KEIL%" (
    echo Cannot find Keil UV4.exe. Please set KEIL_UV4 first.
    echo Example: set KEIL_UV4=YOUR_KEIL_PATH\UV4\UV4.exe
    pause
    exit /b 1
)
if not defined PROJ (
    echo Cannot find BH-F103.uvprojx under Project\RVMDK*
    pause
    exit /b 1
)
start "" "%KEIL%" "%PROJ%"
