@echo off
title Autonomous Trading Agent - Live Watchdog
echo ========================================================
echo   AUTONOMOUS FINANCE AGENT - UNATTENDED 24/7 RUNNER
echo ========================================================
cd /d "%~dp0"
python watchdog_daemon.py
pause
