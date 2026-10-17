@echo off
REM   run_tests.bat              all tests (e2e tests skip if the AI stack is missing)
REM   run_tests.bat -m unit      only unit tests (also: integration, e2e)
cd /d "%~dp0"
.venv\Scripts\python.exe -m pytest %*
