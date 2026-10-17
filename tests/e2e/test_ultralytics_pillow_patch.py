"""
Real Ultralytics: confirm the root cause still exists in the pinned version
and that the scanner is unaffected. Needs Ultralytics, not the model weights.
"""
import pytest
from PIL import Image

pytest.importorskip("ultralytics")
from ultralytics import YOLO  # noqa: E402,F401  (importing applies the patch)

from scanner.photo_scanner import UnreadableImageError, read_photo_info  # noqa: E402
from utils.pil_compat import pillow_open, status  # noqa: E402


def test_root_cause_still_present_in_pinned_ultralytics(tmp_path):
    """If this fails after an Ultralytics upgrade, the workaround may no longer be needed."""
    assert Image.open.__module__ != "PIL.Image"
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ModuleNotFoundError):
        Image.open(bad)


def test_scanner_unaffected_by_real_patch(tmp_path):
    assert pillow_open().__module__ == "PIL.Image"
    assert "saved original" in status()
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(UnreadableImageError, match="UnidentifiedImageError"):
        read_photo_info(bad)
