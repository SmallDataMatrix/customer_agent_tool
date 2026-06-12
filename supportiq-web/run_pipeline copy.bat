@echo off
echo ==============================================
echo SupportIQ Web - Full Pipeline Execution
echo ==============================================

:: Set virtual environment paths
set "VENV_PATH=D:\kw\vt\.venv"
set "PYTHON_EXE=%VENV_PATH%\Scripts\python.exe"
set "STREAMLIT_EXE=%VENV_PATH%\Scripts\streamlit.exe"

:: Step 1: Download data
echo.
echo Step 1/4: Downloading Kaggle Dataset...
%PYTHON_EXE% scripts\01_download_kaggle_data.py
if %errorlevel% neq 0 (
    echo Error: Failed to download data
    exit /b 1
)