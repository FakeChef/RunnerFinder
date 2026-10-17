@echo off
rem Incremental analysis with the NVIDIA GPU, using the separate GPU environment.
rem Same arguments as "index_cli.py analyze"; --ocr-device gpu is added. Example:
rem   analyze_gpu.bat "D:\RacePhotos\Race" --event "2026 Race"
rem Only new, changed and failed photos are analysed; an interrupted run continues.
rem The CPU environment (.venv) is not used or changed. Override the GPU Python with RPF_GPU_PYTHON.
setlocal
if "%RPF_GPU_PYTHON%"=="" set "RPF_GPU_PYTHON=%~dp0.venv-gpu\Scripts\python.exe"
if not exist "%RPF_GPU_PYTHON%" (
  echo GPU environment not found: %RPF_GPU_PYTHON%
  exit /b 2
)
"%RPF_GPU_PYTHON%" "%~dp0index_cli.py" analyze %* --ocr-device gpu
