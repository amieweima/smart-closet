@echo off
title Smart Closet

cd /d "%~dp0"

echo Starting Smart Closet...
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: The .venv virtual environment was not found.
    echo.
    echo Run this command in the VS Code terminal first:
    echo py -m venv .venv
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import streamlit, pandas" >nul 2>&1

if errorlevel 1 (
    echo Installing required packages...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt

    if errorlevel 1 (
        echo.
        echo ERROR: Package installation failed.
        pause
        exit /b 1
    )
)

echo Opening the Streamlit app...
".venv\Scripts\python.exe" -m streamlit run "streamlit_prototype\app.py"

echo.
echo Smart Closet has stopped.
pause