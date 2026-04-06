@echo off
echo Starting YouTube Virality Predictor...
echo Ensure you have installed the requirements: pip install -r ../requirements.txt 
echo.

cd /d "%~dp0\..\src"
python server.py

pause
