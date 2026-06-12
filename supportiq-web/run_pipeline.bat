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

:: Step 2: ETL Pipeline
echo.
echo Step 2/4: Running ETL Pipeline...
%PYTHON_EXE% scripts\02_etl_pipeline.py
if %errorlevel% neq 0 (
    echo Error: ETL Pipeline failed
    exit /b 1
)

:: Step 3: Train ML Models
echo.
echo Step 3/4: Training ML Models...
%PYTHON_EXE% scripts\03_train_models.py
if %errorlevel% neq 0 (
    echo Error: Model training failed
    exit /b 1
)

:: Step 4: Launch Streamlit App
echo.
echo Step 4/4: Launching Streamlit App...
%STREAMLIT_EXE% run app.py

echo.
echo Pipeline completed successfully!
