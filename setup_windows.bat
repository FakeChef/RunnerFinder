@echo off
REM ==================================================================
REM  Runner Photo Finder - Windows setup (reproducible, hash-locked)
REM
REM    setup_windows.bat          full install (scanner + UI + AI stack)
REM    setup_windows.bat core     scanner, database and tests only (small)
REM
REM  Creates .venv with Python 3.11 and installs exactly the versions in
REM  the lock file. pip refuses any file whose hash does not match.
REM ==================================================================
setlocal
cd /d "%~dp0"

set "LOCK=requirements-lock-win-py311.txt"
set "CHECK_ARGS="
if /i "%~1"=="core" (
    set "LOCK=requirements-core-lock-win-py311.txt"
    set "CHECK_ARGS=--core"
)

REM --- 1. Python 3.11, 64-bit -------------------------------------------
py -3.11 -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 64-bit Python 3.11 was not found.
    echo         Install it from https://www.python.org/downloads/windows/
    echo         Choose the 64-bit Windows installer for Python 3.11.x
    echo         and keep the option "py launcher" ticked.
    goto :fail
)

REM --- 2. Virtual environment --------------------------------------------
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)"
    if errorlevel 1 (
        echo [ERROR] .venv exists but is not Python 3.11. Delete the .venv folder and run again.
        goto :fail
    )
) else (
    echo Creating .venv with Python 3.11 ...
    py -3.11 -m venv .venv || goto :fail
)

REM --- 3. Install exactly the locked versions ----------------------------
echo.
echo Installing from %LOCK% ...
if not defined CHECK_ARGS echo The full AI stack is a 1-2 GB download; this can take a while.
.venv\Scripts\python.exe -m pip install --require-hashes --no-deps -r %LOCK% || goto :fail

REM --- 4. Verify ---------------------------------------------------------
echo.
.venv\Scripts\python.exe -m pip check || goto :fail
.venv\Scripts\python.exe check_environment.py %CHECK_ARGS% || goto :fail

echo.
echo Setup finished successfully.
if not defined CHECK_ARGS echo Next: .venv\Scripts\python.exe check_environment.py --models
pause
exit /b 0

:fail
echo.
echo [ERROR] Setup did not complete. See the first error above.
pause
exit /b 1
