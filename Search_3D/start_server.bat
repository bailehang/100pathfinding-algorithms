@echo off
setlocal

cd /d "%~dp0"
set "PORT=5173"
set "URL=http://127.0.0.1:%PORT%/"

echo.
echo 3D Swarm Pathfinding Lab
echo Directory: %cd%
echo URL: %URL%
echo.

where powershell >nul 2>nul
if not %errorlevel% equ 0 (
  echo Could not find Windows PowerShell.
  echo Please run this on Windows, or use another static file server.
  pause
  goto :end
)

echo Starting local server with Windows PowerShell...
echo Keep the server window open while using the page.
echo.

start "3D Swarm Pathfinding Lab Server" /min powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_server.ps1" -Port %PORT%
call :wait_for_server
if %errorlevel% equ 0 (
  start "" "%URL%"
) else (
  echo Server did not start on %URL%
  echo Check the "3D Swarm Pathfinding Lab Server" window for details.
  echo The most common cause is that port %PORT% is already in use.
  pause
)
goto :end

:wait_for_server
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; for($i=0; $i -lt 20; $i++){ try { $c=New-Object Net.Sockets.TcpClient -ArgumentList '127.0.0.1', %PORT%; $c.Close(); $ok=$true; break } catch { Start-Sleep -Milliseconds 250 } }; if($ok){ exit 0 } else { exit 1 }"
exit /b %errorlevel%

:end
endlocal
