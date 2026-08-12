@echo off
chcp 65001 >nul 2>&1
setlocal

where pwsh >nul 2>nul
if %ERRORLEVEL%==0 (
  pwsh -ExecutionPolicy Bypass -File "%~dp0run_dino_vit_compare.ps1" %*
) else (
  powershell -ExecutionPolicy Bypass -File "%~dp0run_dino_vit_compare.ps1" %*
)

exit /b %ERRORLEVEL%
