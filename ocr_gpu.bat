@echo off
rem OCR with the NVIDIA GPU, using the separate GPU environment.
rem Same arguments as "index_cli.py ocr"; --ocr-device gpu is added. Example:
rem   ocr_gpu.bat "D:\RacePhotos\Race" --mode lite1 --limit 0 --no-debug --event "2026 Race"
rem The CPU environment (.venv) is not used or changed. Override the GPU Python with RPF_GPU_PYTHON.
setlocal
if "%RPF_GPU_PYTHON%"=="" set "RPF_GPU_PYTHON=%~dp0.venv-gpu\Scripts\python.exe"
if not exist "%RPF_GPU_PYTHON%" (
  echo GPU environment not found: %RPF_GPU_PYTHON%
  exit /b 2
)
"%RPF_GPU_PYTHON%" "%~dp0index_cli.py" ocr %* --ocr-device gpu
