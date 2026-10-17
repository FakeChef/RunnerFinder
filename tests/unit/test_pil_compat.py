from PIL import Image

import utils.pil_compat as pc


def test_saved_original_is_pillows_own():
    assert pc._ORIGINAL_OPEN is not None
    assert pc._ORIGINAL_OPEN.__module__ == "PIL.Image"


def test_pillow_open_prefers_genuine_function(monkeypatch):
    def wrapper(*a, **k):
        raise AssertionError("wrapper must not be used")
    wrapper.__module__ = "somebody.else"
    monkeypatch.setattr(Image, "open", wrapper)
    assert pc.pillow_open() is pc._ORIGINAL_OPEN
    assert "saved original" in pc.status()


def test_falls_back_to_current_open_when_nothing_saved(monkeypatch):
    def wrapper(*a, **k):
        pass
    wrapper.__module__ = "somebody.else"
    monkeypatch.setattr(Image, "open", wrapper)
    monkeypatch.setattr(pc, "_ORIGINAL_OPEN", None)
    assert pc.pillow_open() is wrapper
    assert "before it could be saved" in pc.status()
