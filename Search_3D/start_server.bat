@echo off
setlocal

cd /d "%~dp0"
set "PORT=5173"
set "CODEX_PYTHON=C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

echo.
echo 3D Swarm Pathfinding Lab
echo Directory: %cd%
echo URL: http://127.0.0.1:%PORT%/
echo.

start "" "http://127.0.0.1:%PORT%/"

if exist "%CODEX_PYTHON%" (
  "%CODEX_PYTHON%" -m http.server %PORT% --bind 127.0.0.1
  goto :end
)

where python >nul 2>nul
if %errorlevel% equ 0 (
  python -m http.server %PORT% --bind 127.0.0.1
  goto :end
)

echo Could not find Python.
echo Install Python or run with a valid python.exe path.
pause

:end
endlocal
