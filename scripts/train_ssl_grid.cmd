@echo off
setlocal

where pwsh >nul 2>nul
if %ERRORLEVEL%==0 (
  pwsh -ExecutionPolicy Bypass -File "%~dp0train_ssl_grid.ps1" %*
) else (
  powershell -ExecutionPolicy Bypass -File "%~dp0train_ssl_grid.ps1" %*
)

exit /b %ERRORLEVEL%
