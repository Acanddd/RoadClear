@echo off

cd /d "%~dp0"

docker compose up --build -d

if errorlevel 1 exit /b 1

echo RoadClear: http://127.0.0.1:5173
