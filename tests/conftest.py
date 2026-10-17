"""
Shared test setup.

Test categories (selected with -m):
  unit         tests/unit/          pure functions
  integration  tests/integration/   real files + real SQLite, no AI
  e2e          tests/e2e/           full AI pipeline (skipped if AI stack missing)

Accuracy on real race photos is NOT measured by these tests; that is
the separate benchmark in evaluation/ (Phase 8).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402,F401  - saves Pillow's Image.open before any test imports Ultralytics
from PIL import Image  # noqa: E402

def pytest_collection_modifyitems(config, items):
    """Mark tests by folder. e2e tests skip themselves if the model is missing."""
    for item in items:
        parts = Path(str(item.fspath)).parts
        for kind in ("unit", "integration", "e2e"):
            if kind in parts:
                item.add_marker(getattr(pytest.mark, kind))


def make_image(path: Path, size=(400, 200), fmt="JPEG", orientation=None,
               datetime_original=None, subsec=None, offset=None,
               make=None, model=None, serial=None, color=(200, 30, 30)) -> Path:
    """
    Write a real image file with real EXIF tags. The top-left quarter is a
    different colour, so tests can check how orientation was applied.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, (240, 240, 240))
    img.paste(color, (0, 0, size[0] // 2, size[1] // 2))
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    if make:
        exif[0x010F] = make
    if model:
        exif[0x0110] = model
    sub = exif.get_ifd(0x8769)
    if datetime_original:
        sub[0x9003] = datetime_original
    if subsec is not None:
        sub[0x9291] = subsec
    if offset:
        sub[0x9011] = offset
    if serial:
        sub[0xA431] = serial
    img.save(path, fmt, exif=exif.tobytes())
    return path


@pytest.fixture
def image_factory():
    return make_image
