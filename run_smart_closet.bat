@echo off
setlocal

title Smart Closet Launcher

set "PROJECT_DIR=C:\Users\nomno\OneDrive\Desktop\smart-closet"
set "FRONTEND_DIR=%PROJECT_DIR%\frontend"
set "VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"

echo ============================================
echo        Starting Smart Closet
echo ============================================
echo.

if not exist "%PROJECT_DIR%" (
    echo ERROR: Project folder was not found:
    echo %PROJECT_DIR%
    echo.
    pause
    exit /b 1
)

if not exist "%FRONTEND_DIR%" (
    echo ERROR: Frontend folder was not found:
    echo %FRONTEND_DIR%
    echo.
    pause
    exit /b 1
)

where npm.cmd >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm was not found.
    echo Install Node.js or restart your computer after installing it.
    echo.
    pause
    exit /b 1
)

if exist "%VENV_PYTHON%" (
    set "PYTHON_CMD=%VENV_PYTHON%"
) else (
    where python.exe >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python was not found.
        echo Expected virtual environment:
        echo %VENV_PYTHON%
        echo.
        pause
        exit /b 1
    )
    set "PYTHON_CMD=python"
)

echo Starting backend...
start "Smart Closet Backend" cmd /k "cd /d ""%PROJECT_DIR%"" && ""%PYTHON_CMD%"" -m uvicorn backend.app.main:app --reload --port 8000"

timeout /t 3 /nobreak >nul

echo Starting frontend...
start "Smart Closet Frontend" cmd /k "cd /d ""%FRONTEND_DIR%"" && npm.cmd run dev"

echo Waiting for the app to start...
timeout /t 5 /nobreak >nul

echo Opening Smart Closet...
start "" "http://localhost:5173/"

echo.
echo Smart Closet has been launched.
echo Keep the two terminal windows open while using the app.
echo You may close this launcher window.
echo.
pause
