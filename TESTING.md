# Testing

There are two suites:

- `tests/` uses **pytest**, with markers `unit`, `integration` and `e2e` (`pytest.ini`).
- `desktop/tests/` uses **unittest** and covers the PySide6 GUI and the analysis worker.

These tests show that the code behaves as designed. **They do not measure how well bibs are recognised in real race photos.** Accuracy is evaluated separately with your own labelled photos (`evaluation/`).

## Engine tests (pytest)

| Marker | Folder | Needs | Writes |
|---|---|---|---|
| `unit` | `tests/unit` | Pillow, PyYAML, pytest (`setup_windows.bat core`) | pytest temporary folders only |
| `integration` | `tests/integration` | same as unit | temporary image files and SQLite databases; no models |
| `e2e` | `tests/e2e` | full engine install plus the models ([MODEL_SETUP.md](MODEL_SETUP.md)); skipped automatically when the models are missing | temporary folders |

```
run_tests.bat
run_tests.bat -m unit
run_tests.bat -m integration
```

`run_tests.bat` runs `.venv\Scripts\python.exe -m pytest` with your arguments.

## Desktop tests (unittest)

Run them with the desktop environment, from the repository folder, without a display:

```
.venv-desktop\Scripts\python.exe -m unittest desktop.tests.test_skeleton desktop.tests.test_v12_search desktop.tests.test_v13_gui -v
```

These three build their data in temporary folders. They set `QT_QPA_PLATFORM=offscreen` themselves.

The other desktop tests need **private data** that is not in the repository: a reference photo set, a populated index, a real event folder, the video evidence, a RunnerFinder 1.2 checkout and index, and a GPU worker Python. Each item is read from an environment variable (`desktop/tests/private_data.py`), and a test that needs a missing item is **skipped** with the variable's name:

| Variable | Item |
|---|---|
| `RUNNERFINDER_TEST_PHOTOS` | the 370-photo reference set (the tests assert this exact count) |
| `RUNNERFINDER_TEST_INDEX` | a populated index; default `data\runner_photo_finder.db`. It is copied and never written, and a `thumbnails` folder next to it is copied too. |
| `RUNNERFINDER_REAL_EVENT` | a full event folder (counted only) |
| `RUNNERFINDER_BLIND_LABELS` | blind-label CSV (optional extra search bibs) |
| `RUNNERFINDER_VIDEO_EVIDENCE` | the recorded video feasibility evidence |
| `RUNNERFINDER_V12_SOURCE`, `RUNNERFINDER_V12_INDEX` | a RunnerFinder 1.2 checkout and index, for the compatibility test |
| `RUNNERFINDER_WORKER_PYTHON` | GPU engine Python for the real-engine tests; default `.venv-gpu` |

Several of these tests compare against figures of that specific data set (photo counts, timings, an RTX GPU name). With other data they are expected to fail.

## Keeping the tree clean

pytest writes `.pytest_cache` and Python writes `__pycache__` next to the sources. To avoid both:

```
set PYTHONDONTWRITEBYTECODE=1
.venv\Scripts\python.exe -m pytest -m unit -p no:cacheprovider
```

Loading the configuration creates an empty `data\` folder if none exists (`config.py`, `get_config`).
