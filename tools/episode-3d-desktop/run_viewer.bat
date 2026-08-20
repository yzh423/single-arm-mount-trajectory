@echo off
setlocal
cd /d "%~dp0\..\.."
python scripts\episode_3d_desktop.py %*
if errorlevel 1 pause
