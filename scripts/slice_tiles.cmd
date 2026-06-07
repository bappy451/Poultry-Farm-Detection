@echo off
setlocal

where pwsh >nul 2>nul
if %ERRORLEVEL%==0 (
  pwsh -ExecutionPolicy Bypass -File "%~dp0slice_tiles.ps1" %*
) else (
  powershell -ExecutionPolicy Bypass -File "%~dp0slice_tiles.ps1" %*
)

exit /b %ERRORLEVEL%
