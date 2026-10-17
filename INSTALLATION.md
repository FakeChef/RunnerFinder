# Installation (from source)

RunnerFinder uses up to three separate Python environments, all **Python 3.11, 64-bit, Windows**:

| Environment | Folder (suggested) | Contents | Lock file |
|---|---|---|---|
| Engine (CPU) | `.venv` | scanner, YOLO, PaddleOCR (CPU), legacy Streamlit UI, pytest | `requirements-lock-win-py311.txt` |
| Desktop GUI | `.venv-desktop` | PySide6 and Pillow only | `requirements-desktop-lock-win-py311.txt` |
| Engine (GPU) | `.venv-gpu` | PaddlePaddle GPU build and NVIDIA CUDA/cuDNN wheels | none yet (see below) |

The desktop GUI never imports the AI stack. It starts the analysis in a separate *worker* Python (see "Desktop app" below).

All commands work in Command Prompt and PowerShell. They call each environment's Python directly, so nothing needs to be "activated".

## 1. Python 3.11

Install the 64-bit Windows installer of Python 3.11 from <https://www.python.org/downloads/windows/> with the **py launcher** option. Check it:

```
py -3.11 --version
```

## 2. Engine environment (`.venv`)

From the repository folder:

```
setup_windows.bat
```

The script creates `.venv` with Python 3.11 and installs exactly the versions in `requirements-lock-win-py311.txt` (`pip install --require-hashes --no-deps`, so every file's SHA-256 is checked). It then runs `pip check` and `check_environment.py`.

To install only what the unit and integration tests need (Pillow, PyYAML and pytest), without AI:

```
setup_windows.bat core
```

Next, download the models: see [MODEL_SETUP.md](MODEL_SETUP.md).

## 3. Desktop GUI environment (`.venv-desktop`)

Not scripted yet. Create it manually:

```
py -3.11 -m venv .venv-desktop
.venv-desktop\Scripts\python.exe -m pip install --require-hashes --no-deps -r requirements-desktop-lock-win-py311.txt
```

Start the app without a console window:

```
.venv-desktop\Scripts\pythonw.exe desktop\RunnerFinder.pyw
```

With a console instead: `.venv-desktop\Scripts\python.exe -m desktop`, run from the repository folder.

**Worker Python.** The desktop app runs the analysis in the first of these:
1. the `RUNNERFINDER_WORKER_PYTHON` environment variable,
2. `.venv-gpu\Scripts\python.exe` inside the repository,
3. the GUI's own Python. That one has no AI stack, so analysis will not work with it.

The New Event page checks the GPU through the worker and enables **Start** only when PaddlePaddle can use an NVIDIA GPU. Searching an existing index works without a GPU.

> This procedure has not yet been verified from a clean checkout.

## 4. GPU engine environment (`.venv-gpu`), optional

GPU OCR uses a separate environment with `paddlepaddle-gpu` 3.3.1 and the NVIDIA CUDA 13 / cuDNN 9 wheels. Only OCR runs on the GPU; person detection stays on the CPU.

**A reproducible, hash-locked GPU setup is not available yet.** `requirements-gpu-win-py311.txt` records the exact package versions of the tested GPU environment (Python 3.11.9, RTX 3060, CUDA 13.0, cuDNN 9.13). It is a record, not a lock file, and it does not name the package index that provides `paddlepaddle-gpu`.

Once a GPU environment exists:

```
<gpu python> index_cli.py gpu-check
```

The batch files `analyze_gpu.bat` and `ocr_gpu.bat` use `RPF_GPU_PYTHON`, or `.venv-gpu\Scripts\python.exe` when it is not set.

`--ocr-device` (command line):
- `cpu` (the default),
- `gpu`: uses the GPU or stops with an error,
- `auto`: uses the GPU if it works, otherwise the CPU, and prints the reason.

## Where data is stored

Everything is under `data\` in the repository folder:

| What | Where |
|---|---|
| index | `data\runner_photo_finder.db` |
| thumbnails | `data\thumbnails\` |
| diagnostic output | `data\debug\` |

The model files are under `models\`. Neither folder is tracked by Git.

## Troubleshooting

- **"64-bit Python 3.11 was not found"** (from `setup_windows.bat`): install it with the py launcher option and open a new terminal.
- **Hash mismatch during setup:** a download was corrupted or altered. Run the setup again.
- **OpenCV broken after removing a package:** `opencv-python` and `opencv-contrib-python` share one `cv2` folder and are pinned to the same version. Never uninstall one on its own. `check_environment.py` detects the problem and prints the repair command:
  ```
  .\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps --require-hashes -r requirements-opencv-repair-win-py311.txt
  ```
