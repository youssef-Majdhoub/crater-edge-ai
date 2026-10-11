@echo off
REM ==========================================================================
REM  C.R.A.T.E.R. - LIVE link (real edge_detector.py -> hazards.json -> radio -> dashboard)
REM  1) run this file   2) in another terminal run:  python edge_detector.py test.jpg
REM  Do NOT run it together with run_pipeline.bat (both use UDP port 9002).
REM ==========================================================================
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    pause
    exit /b 1
)

python -c "import streamlit, plotly, pandas, reedsolo" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    python -m pip install -r requirements.txt "streamlit>=1.37" plotly pandas
)

if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    mkdir "%USERPROFILE%\.streamlit" 2>nul
    > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo [1/3] Starting Mission Control dashboard...
start "CRATER - Dashboard" cmd /k python -m streamlit run dashboard.py --server.headless true --server.port 8501 --browser.gatherUsageStats false
timeout /t 5 /nobreak >nul

echo [2/3] Opening browser...
start "" http://localhost:8501
timeout /t 4 /nobreak >nul

echo [3/3] Starting rover radio (30%% simulated packet loss)...
start "CRATER - Telemetry TX" cmd /k python telemetry_tx.py

echo.
echo Link is up. Now run in this folder:   python edge_detector.py test.jpg
endlocal
