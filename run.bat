@echo off
title CogniArousal - Responsible AI Cognitive Analysis Platform

echo.
echo  ============================================================
echo   COGNIAROUSAL  v1.0  --  Responsible AI Platform
echo   IEEE EMBS Research Prototype
echo  ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install Python 3.11+ from python.org
    pause
    exit /b 1
)

:: Install dependencies
echo  [1/3] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

:: Check if models exist, run training if not
if not exist "models\arousal\model.pkl" (
    echo.
    echo  [2/3] No trained models found. Running data pipeline...
    if not exist "output\feature_dataset.csv" (
        echo        Running Part 1: Data Processing...
        python main.py --data_dir data/ --output_dir output/
        if errorlevel 1 (
            echo  [ERROR] Data processing failed. Ensure WESAD data is in data/
            pause
            exit /b 1
        )
    )
    echo        Running Part 2: Model Training...
    python train.py --feature_csv output/feature_dataset.csv --models_dir models/
    if errorlevel 1 (
        echo  [ERROR] Model training failed.
        pause
        exit /b 1
    )
) else (
    echo  [2/3] Models found. Skipping training.
)

:: Launch dashboard
echo.
echo  [3/3] Launching CogniArousal Dashboard...
echo.
echo  Open your browser at: http://localhost:8501
echo.

streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

pause
