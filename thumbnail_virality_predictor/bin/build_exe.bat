@echo off
echo Starting YouTube Virality Predictor Build Process...
echo This will package the backend, AI models, and frontend into a standalone executable.
echo Please securely keep your global python environment running (C:\Python314\python.exe).
echo.
echo Building... (This may take 10-15 minutes due to AI libraries).

cd /d "%~dp0\.."
C:\Python314\python.exe -m PyInstaller bin\YouTubeViralityPredictor.spec --clean --noconfirm

echo.
echo Build Complete!
echo You can find your executable inside the "dist\" folder.
pause
