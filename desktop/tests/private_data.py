"""Optional private test data for the desktop tests.

None of this data is in the repository. Each path comes from an environment variable;
a test that needs it is skipped when the variable is not set or the path does not exist.

  RUNNERFINDER_TEST_PHOTOS     folder of the 370-photo validation set (read-only; the tests assert that count)
  RUNNERFINDER_REAL_EVENT      a full event folder: 6,974 photos, a C2 card and a V folder of 53 clips (counted only)
  RUNNERFINDER_BLIND_LABELS    blind_labels.csv of that event's validation (optional extra search bibs)
  RUNNERFINDER_TEST_INDEX      a populated index, default <checkout>/data/runner_photo_finder.db; copied, never
                               written. Thumbnails are copied from the "thumbnails" folder next to it.
  RUNNERFINDER_VIDEO_EVIDENCE  the B1 video feasibility folder (data/ai_evidence.jsonl, data/b2_per_video.csv)
  RUNNERFINDER_V12_SOURCE      source folder of RunnerFinder 1.2 (compatibility test)
  RUNNERFINDER_V12_INDEX       a RunnerFinder 1.2 index, opened read-only (compatibility test)
  RUNNERFINDER_WORKER_PYTHON   engine Python for the real-engine tests, default <checkout>/.venv-gpu as in the app
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path(variable: str, default: Path | None = None) -> Path | None:
    value = os.environ.get(variable, "").strip()
    return Path(value) if value else default


TEST_PHOTOS = _path("RUNNERFINDER_TEST_PHOTOS")
REAL_EVENT = _path("RUNNERFINDER_REAL_EVENT")
BLIND_LABELS = _path("RUNNERFINDER_BLIND_LABELS")
INDEX_DB = _path("RUNNERFINDER_TEST_INDEX", ROOT / "data" / "runner_photo_finder.db")
THUMBNAILS = INDEX_DB.parent / "thumbnails"
VIDEO_EVIDENCE = _path("RUNNERFINDER_VIDEO_EVIDENCE")
V12_SOURCE = _path("RUNNERFINDER_V12_SOURCE")
V12_INDEX = _path("RUNNERFINDER_V12_INDEX")
#: same fallback as desktop.services.analysis_job.GPU_VENV_PYTHON (not imported here: that module needs Qt)
ENGINE_PYTHON = _path("RUNNERFINDER_WORKER_PYTHON", ROOT / ".venv-gpu" / "Scripts" / "python.exe")


def available(path: Path | None) -> bool:
    return path is not None and path.exists()


def needs(path: Path | None, variable: str):
    """Skip decorator for a test or test class that needs one item of private data."""
    return unittest.skipUnless(available(path), f"{variable} not set or not found")
