# RunnerFinder

Find race photos and video clips by bib number (`10258`, `T1234`, `UT001`) on your own Windows PC.

RunnerFinder scans a folder of event photos and videos, finds the runners with a person
detector, reads their bib numbers with OCR, and stores the results in a local SQLite index
that you can search by bib. Photos are never uploaded, and the originals are only read,
never modified.

Version: **1.3.1** (`desktop/__init__.py`)

## What is in this repository

| Part | Start | What it does |
|---|---|---|
| Desktop app (PySide6) | `pythonw desktop\RunnerFinder.pyw` or `python -m desktop` | Dashboard, New Event (choose a folder and analyse it), Analysis Progress, Search |
| Command line | `python index_cli.py <command>` | `scan`, `analyze`, `thumbs`, `filter-bibs`, `import-csv`, `detect`, `ocr`, `benchmark`, `gpu-check` |
| Legacy web UI (Streamlit) | `run_app.bat` | Earlier search interface, reachable only from this PC (`localhost`) |

Recognition pipeline:
- Person detection: Ultralytics **YOLO26n**.
- Bib OCR: **PaddleOCR** with the PP-OCRv5 mobile models (`PP-OCRv5_mobile_det`, `en_PP-OCRv5_mobile_rec`).
- A rule-based noise filter (`config/bib_rules.yaml`) hides numbers that are not bibs. Nothing is deleted.

Supported files: photos `.jpg`, `.jpeg`, `.png`; videos `.mp4`, `.mov`, `.m4v` (case-insensitive).

## Requirements

- Windows, 64-bit, and **Python 3.11** (64-bit). The hash-locked dependency files are built for exactly this platform.
- Analysis from the **desktop app** runs OCR on an **NVIDIA GPU** and is disabled when no GPU environment is found. Searching works without a GPU. The command line can also run OCR on the CPU (`--ocr-device cpu`, the default).
- About 1-2 GB of downloads for the AI packages, plus the model files (see [MODEL_SETUP.md](MODEL_SETUP.md)).

Tested on: Windows 10 (x64), Python 3.11.9, NVIDIA RTX 3060 with CUDA 13.0 and cuDNN 9.13. This is a tested configuration, not a minimum requirement.

## Quick start (command line, CPU)

```
setup_windows.bat
.\.venv\Scripts\python.exe check_environment.py --models
.\.venv\Scripts\python.exe index_cli.py analyze "D:\RacePhotos\MyRace2026" --event "My Race 2026"
```

For the desktop app and the GPU environment, see [INSTALLATION.md](INSTALLATION.md).

## Documentation

| Document | Content |
|---|---|
| [INSTALLATION.md](INSTALLATION.md) | Setup of the engine, desktop and GPU environments |
| [MODEL_SETUP.md](MODEL_SETUP.md) | Downloading the YOLO and PaddleOCR models, offline use |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Code layout, configuration, environment variables, dependency files |
| [TESTING.md](TESTING.md) | Test suites, what each needs, how to run them safely |

## Privacy and network use

- Photos and videos never leave the PC.
- Network access is needed only to install packages and to download the models once.
- Once `models\yolo26n.pt` exists, Ultralytics runs offline (`YOLO_OFFLINE=1`, set in `config.py`). Its usage analytics are switched off (`sync=False`, `detection/person_detector.py`). An e2e test blocks all network connections during detection and fails if any are attempted.
- The Streamlit UI listens on `localhost` only and has its usage statistics off (`.streamlit/config.toml`).

## Bib numbers

Bibs are always handled as **text**, so leading zeros are kept (`UT001`, `00123`). OCR results, search input and manual corrections are normalised the same way: upper-case, with whitespace and the separators `- _ . # · : /` removed (`ocr/bib_format.py`). Length and pattern rules are in `config.py` → `BibFormatConfig`.

## License

Copyright (C) 2026 the RunnerFinder authors.

RunnerFinder is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see [LICENSE](LICENSE) for details.

Third-party packages and models are not included in this repository and keep their own licenses: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
