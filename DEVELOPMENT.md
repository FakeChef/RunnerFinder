# Development

## Code layout

| Path | Content |
|---|---|
| `config.py` | All settings as dataclasses (`AppConfig`, `PathsConfig`, `DetectionConfig`, `OCRConfig`, `BibFormatConfig`, `ScanConfig`, `ThumbnailConfig`, `DebugConfig`); sets the model-cache and offline environment variables before any AI import |
| `scanner/` | Find photo files and read metadata (`photo_scanner.py`); incremental folder sync (`folder_sync.py`); prefetch and the three-stage analysis pipeline (`prefetch.py`, `pipeline3.py`) |
| `detection/` | Person detection (YOLO26n), geometry, ranking for the Lite modes, debug images |
| `ocr/` | OCR engine wrapper (PaddleOCR), bib candidates and format rules, per-photo OCR, noise filter, diagnostics |
| `indexing/` | Incremental, resumable analysis of a folder into an event; re-scoring stored candidates with the noise filter |
| `video/` | Video frames, video analysis, the video bib decision rules |
| `database/` | SQLite access layer, schema (`schema.sql`, schema version 3), video extension (`video_schema.sql`), migrations, CSV import |
| `search/` | Bib search on the index (photos and videos) |
| `export/` | Copy matching photos into an export folder layout |
| `utils/` | Thumbnails, GPU runtime helpers, Pillow compatibility, XLSX writer, system info |
| `evaluation/` | Accuracy evaluation against your own labelled photos |
| `desktop/` | PySide6 GUI: `app.py` (entry point), `main_window.py`, `pages/`, `services/` (search service, analysis job and worker, progress protocol), `resources/` (icons, drawn by `make_icon.py`) |
| `index_cli.py` | Command line (`python index_cli.py --help`) |
| `app.py`, `run_app.bat` | Legacy Streamlit UI |
| `check_environment.py` | Checks installed versions against the lock file, imports, OpenCV and GPU; `--models` downloads and runs the models |

### Process model of the desktop app
The GUI process never imports the AI stack (PaddlePaddle, PaddleOCR, PyTorch, Ultralytics, OpenCV, `ocr`, `detection`); `desktop/tests/test_analysis_gui.py` checks this. The analysis runs in a separate worker process (`desktop/services/analysis_worker.py`), started through `QProcess` by `desktop/services/analysis_job.py`. The worker reports progress as JSON lines (`desktop/services/progress_protocol.py`) and stops when its stdin closes.

## Environment variables

| Variable | Used by | Effect |
|---|---|---|
| `RUNNERFINDER_WORKER_PYTHON` | desktop app | Python for the analysis worker (default: `.venv-gpu`, else the GUI's own Python) |
| `RPF_GPU_PYTHON` | `analyze_gpu.bat`, `ocr_gpu.bat` | GPU Python (default: `.venv-gpu`) |
| `PADDLE_PDX_CACHE_HOME` | PaddleOCR | Model cache; `config.py` sets it to `models\paddlex` unless already set |
| `PADDLE_PDX_MODEL_SOURCE` | PaddleOCR | Alternative model download source (for example `bos`) |
| `YOLO_OFFLINE`, `YOLO_AUTOINSTALL`, `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` | Ultralytics, PaddleOCR | Set by `config.py` with `setdefault`, so an existing value wins |
| `RUNNERFINDER_TEST_*`, `RUNNERFINDER_REAL_EVENT`, `RUNNERFINDER_BLIND_LABELS`, `RUNNERFINDER_VIDEO_EVIDENCE`, `RUNNERFINDER_V12_*` | desktop tests | Optional private test data, see [TESTING.md](TESTING.md) |

## Dependency files

| File | Purpose |
|---|---|
| `requirements-engine.txt` | Direct engine dependencies (scanner, YOLO, PaddleOCR, OpenCV) |
| `requirements-desktop.txt` | Direct desktop GUI dependencies (PySide6, Pillow) |
| `requirements-dev.txt` | Direct test dependencies (Pillow, PyYAML, pytest) |
| `requirements-ui.txt` | Legacy Streamlit UI |
| `requirements.txt` | Engine + UI + dev together |
| `requirements-lock-win-py311.txt` | Hash lock of `requirements.txt` (used by `setup_windows.bat`) |
| `requirements-core-lock-win-py311.txt` | Hash lock of `requirements-dev.txt` (used by `setup_windows.bat core`) |
| `requirements-desktop-lock-win-py311.txt` | Hash lock of `requirements-desktop.txt` |
| `requirements-opencv-repair-win-py311.txt` | Hash-pinned OpenCV repair |
| `requirements-gpu-win-py311.txt` | Version record of the tested GPU environment (not a lock) |

Edit only the direct files, then regenerate the locks **from the repository folder**:

```
uv pip compile requirements.txt         --python-platform x86_64-pc-windows-msvc --python-version 3.11 --generate-hashes --exclude-newer 2026-09-29T00:00:00Z --no-header -o requirements-lock-win-py311.txt
uv pip compile requirements-dev.txt     --python-platform x86_64-pc-windows-msvc --python-version 3.11 --generate-hashes --exclude-newer 2026-09-29T00:00:00Z --no-header -o requirements-core-lock-win-py311.txt
uv pip compile requirements-desktop.txt --python-platform x86_64-pc-windows-msvc --python-version 3.11 --generate-hashes --exclude-newer 2026-09-29T00:00:00Z --no-header -o requirements-desktop-lock-win-py311.txt
```

`--exclude-newer` keeps the resolution to packages published before that date. Change it on purpose only.

OpenCV: Ultralytics depends on `opencv-python`, while PaddleOCR checks for `opencv-contrib-python` by name. Both install the same `cv2` folder, so they are pinned to the identical version (4.10.0.84).

## Conventions

- Bibs are text everywhere, so leading zeros are kept.
- All coordinates are full-resolution pixels after EXIF rotation.
- Original photos are only opened for reading. Debug and benchmark output is written only inside `data\`, and tests check both rules.
- Line endings: `.bat` files use CRLF. Keep each existing file's line endings when editing.
