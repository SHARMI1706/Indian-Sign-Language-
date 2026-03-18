@echo off
REM ISL Translator - Quick Start Script (Windows)
REM This script automates the setup and deployment process

echo.
echo ================================
echo ISL Translator - Quick Start
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo OK Python detected
python --version
echo.

REM Check if model file exists
if not exist "isl_digit_svm_model.pkl" (
    echo X Model file 'isl_digit_svm_model.pkl' not found!
    echo Please place your trained model file in the project directory.
    pause
    exit /b 1
)

echo OK Model file found
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo OK Virtual environment created
) else (
    echo OK Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
echo OK Dependencies installed
echo.

REM Start the application
echo ================================
echo Starting ISL Translator...
echo.
echo Server will start at: http://localhost:5000
echo Make sure to allow camera permissions in your browser
echo.
echo Press Ctrl+C to stop the server
echo ================================
echo.

python app.py

pause