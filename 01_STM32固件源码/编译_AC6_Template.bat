@echo off
chcp 65001 >nul
set "KEIL=%KEIL_UV4%"
if "%KEIL%"=="" if exist "C:\Keil_v5\UV4\UV4.exe" set "KEIL=C:\Keil_v5\UV4\UV4.exe"
if "%KEIL%"=="" if exist "C:\Keil\UV4\UV4.exe" set "KEIL=C:\Keil\UV4\UV4.exe"
if "%KEIL%"=="" if exist "%ProgramFiles%\Keil_v5\UV4\UV4.exe" set "KEIL=%ProgramFiles%\Keil_v5\UV4\UV4.exe"
set "PROJ="
set "LOG=%~dp0ac6_build.log"

for /d %%D in ("%~dp0Project\RVMDK*") do (
    if exist "%%~fD\BH-F103.uvprojx" set "PROJ=%%~fD\BH-F103.uvprojx"
)

if not exist "%KEIL%" (
    echo Cannot find Keil UV4.exe. Please set KEIL_UV4 first.
    echo Example: set KEIL_UV4=YOUR_KEIL_PATH\UV4\UV4.exe
    if /i not "%~1"=="nopause" pause
    exit /b 1
)

if not defined PROJ (
    echo Cannot find BH-F103.uvprojx under Project\RVMDK*
    if /i not "%~1"=="nopause" pause
    exit /b 1
)

if exist "%LOG%" del /f /q "%LOG%"
"%KEIL%" -b "%PROJ%" -t "LDC" -j0 -o "%LOG%"
set "BUILD_ERR=%ERRORLEVEL%"
if exist "%LOG%" type "%LOG%"
if /i not "%~1"=="nopause" pause
exit /b %BUILD_ERR%
