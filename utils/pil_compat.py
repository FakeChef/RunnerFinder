"""
Keep access to Pillow's genuine Image.open.

Root cause: importing Ultralytics replaces PIL.Image.open for the whole
program (ultralytics/utils/patches.py, "Image.open = image_open"). When a
file cannot be identified, the replacement tries to install/import the
optional HEIF plugin pi-heif. With auto-install disabled (our setting)
it then raises ModuleNotFoundError instead of Pillow's documented
UnidentifiedImageError, and logs a warning for every such file.

Fix: this module saves a reference to Pillow's own function when it is
first imported. config.py imports it, and config.py is always imported
before Ultralytics (it must be, to switch off Ultralytics' auto-install),
so the saved reference is taken before any patch is applied.

Nothing here depends on Ultralytics' internals. Functions are identified
only by their standard __module__ attribute ("PIL.Image" for Pillow's).
"""
from __future__ import annotations

from PIL import Image

PILLOW_MODULE = "PIL.Image"


def _is_pillow_open(func) -> bool:
    return callable(func) and getattr(func, "__module__", None) == PILLOW_MODULE


# Saved at first import. None if something had already replaced Image.open.
_ORIGINAL_OPEN = Image.open if _is_pillow_open(Image.open) else None


def pillow_open():
    """
    Return Pillow's own Image.open if it can be obtained, otherwise the
    current (replaced) Image.open. Callers must still handle any exception
    from it, because in the last case its error types are unknown.
    """
    if _is_pillow_open(Image.open):
        return Image.open
    if _ORIGINAL_OPEN is not None:
        return _ORIGINAL_OPEN
    return Image.open


def status() -> str:
    """One-line description for check_environment.py."""
    if _is_pillow_open(Image.open):
        return "PIL.Image.open is Pillow's own"
    if _ORIGINAL_OPEN is not None:
        return (f"PIL.Image.open was replaced by {Image.open.__module__}; "
                "the scanner uses Pillow's saved original")
    return (f"PIL.Image.open was replaced by {getattr(Image.open, '__module__', '?')} before it "
            "could be saved; corrupt files are still handled, with less precise messages")
