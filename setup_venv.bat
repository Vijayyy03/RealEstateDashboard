@echo off
REM Batch script to set up a virtual environment with Python 3.11.9

REM Check if Python 3.11.9 is installed
python --version 2>&1 | findstr /C:"3.11.9" > nul
if errorlevel 1 (
    echo Python 3.11.9 is not detected. Please install Python 3.11.9 before running this script.
    echo You can download it from: https://www.python.org/downloads/release/python-3119/
    exit /b 1
)

echo Python 3.11.9 detected. Setting up virtual environment...

REM Create a virtual environment
set venvName=venv-py3119
echo Creating virtual environment: %venvName%
python -m venv %venvName%

REM Activate the virtual environment and install dependencies
echo Activating virtual environment and installing dependencies...

REM Check if activation script exists
if exist "%venvName%\Scripts\activate.bat" (
    call %venvName%\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    
    echo.
    echo Dependencies installed successfully!
    echo.
    echo To activate this environment in the future, run:
    echo %venvName%\Scripts\activate.bat
    echo.
    echo To deactivate the environment, simply run:
    echo deactivate
) else (
    echo Error: Virtual environment activation script not found
    exit /b 1
)