"""Scanner against real image files written to a temporary folder."""
import builtins
import hashlib
import os
import stat
import sys
from pathlib import Path

import pytest
from PIL import Image

from scanner.photo_scanner import (UnreadableImageError, iter_photo_files,
                                   load_image_oriented, read_photo_info)


def names(root, **kw):
    return [os.path.relpath(p, root).replace(os.sep, "/") for p, _ in iter_photo_files(root, **kw)]


# ---------------------------------------------------------------- discovery

def test_recursive_discovery_and_extensions(tmp_path, image_factory):
    image_factory(tmp_path / "a.jpg")
    image_factory(tmp_path / "B.JPEG")
    image_factory(tmp_path / "c.PNG", fmt="PNG")
    image_factory(tmp_path / "sub" / "deeper" / "d.Jpg")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "raw.CR3").write_bytes(b"raw")          # RAW is not supported in V0.1
    assert names(tmp_path) == ["a.jpg", "B.JPEG", "c.PNG", "sub/deeper/d.Jpg"]
    assert names(tmp_path, recursive=False) == ["a.jpg", "B.JPEG", "c.PNG"]


def test_skips_hidden_junk_system_and_export_folders(tmp_path, image_factory):
    image_factory(tmp_path / "keep.jpg")
    image_factory(tmp_path / ".hidden" / "x.jpg")
    image_factory(tmp_path / "._keep.jpg")                # macOS metadata file
    image_factory(tmp_path / "$RECYCLE.BIN" / "y.jpg")
    image_factory(tmp_path / "Export" / "Race" / "Bib_1" / "keep.jpg")
    found = names(tmp_path, skip_dir_names=("$RECYCLE.BIN",),
                  exclude_dirs=(tmp_path / "Export",))
    assert found == ["keep.jpg"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows hidden attribute")
def test_skips_windows_hidden_attribute(tmp_path, image_factory):
    import ctypes
    image_factory(tmp_path / "keep.jpg")
    hidden = image_factory(tmp_path / "hidden.jpg")
    ctypes.windll.kernel32.SetFileAttributesW(str(hidden), 0x2)   # FILE_ATTRIBUTE_HIDDEN
    assert names(tmp_path) == ["keep.jpg"]


def test_symlink_loops_do_not_hang(tmp_path, image_factory):
    image_factory(tmp_path / "a" / "x.jpg")
    try:
        os.symlink(tmp_path, tmp_path / "a" / "loop", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here (Windows needs Developer Mode)")
    assert names(tmp_path) == ["a/x.jpg"]


def test_unicode_and_spaces_in_paths(tmp_path, image_factory):
    p = image_factory(tmp_path / "2026 太湖越野" / "终点 IMG_0001.jpg")
    assert names(tmp_path) == ["2026 太湖越野/终点 IMG_0001.jpg"]
    assert read_photo_info(p).file_name == "终点 IMG_0001.jpg"


# ---------------------------------------------------------------- metadata

def test_full_metadata(tmp_path, image_factory):
    p = image_factory(tmp_path / "a.jpg", datetime_original="2026:04:12 08:31:22",
                      subsec="45", offset="+08:00", make="Canon",
                      model="Canon EOS R5", serial="012345678")
    info = read_photo_info(p)
    assert (info.width, info.height) == (400, 200)
    assert info.capture_time == "2026-04-12T08:31:22.450"
    assert info.capture_offset == "+08:00"
    assert info.capture_time_source == "exif_original"
    assert (info.camera_make, info.camera_model, info.camera_serial) == \
        ("Canon", "Canon EOS R5", "012345678")
    assert info.file_size == p.stat().st_size and len(info.fingerprint) == 64


def test_photo_without_exif(tmp_path, image_factory):
    info = read_photo_info(image_factory(tmp_path / "plain.png", fmt="PNG"))
    assert info.capture_time is None and info.exif_orientation is None
    assert (info.width, info.height) == (400, 200)


# Raw image: 400x200, red marker in the top-left quarter. For each EXIF
# orientation: expected displayed size and the corner where the marker ends up.
ORIENTATIONS = {
    1: ((400, 200), "top-left"),     2: ((400, 200), "top-right"),
    3: ((400, 200), "bottom-right"), 4: ((400, 200), "bottom-left"),
    5: ((200, 400), "top-left"),     6: ((200, 400), "top-right"),
    7: ((200, 400), "bottom-right"), 8: ((200, 400), "bottom-left"),
}


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation(tmp_path, image_factory, orientation):
    size, corner = ORIENTATIONS[orientation]
    p = image_factory(tmp_path / f"o{orientation}.jpg", orientation=orientation)
    info = read_photo_info(p)
    img = load_image_oriented(p)
    # Metadata dimensions and decoded image use the same coordinate system.
    assert (info.width, info.height) == img.size == size
    w, h = img.size
    points = {"top-left": (5, 5), "top-right": (w - 5, 5),
              "bottom-left": (5, h - 5), "bottom-right": (w - 5, h - 5)}
    r, g, b = img.getpixel(points[corner])
    assert r > 150 and g < 100, f"marker not at {corner}"


# ---------------------------------------------------------------- corrupt files

def test_garbage_file_is_unreadable(tmp_path):
    p = tmp_path / "fake.jpg"
    p.write_bytes(b"this is not an image" * 100)
    with pytest.raises(UnreadableImageError):
        read_photo_info(p)


def test_empty_file_is_unreadable(tmp_path):
    p = tmp_path / "empty.jpg"
    p.write_bytes(b"")
    with pytest.raises(UnreadableImageError, match="empty"):
        read_photo_info(p)


def test_truncated_jpeg_fails_safely_on_full_decode(tmp_path):
    big = Image.effect_noise((1200, 800), 64).convert("RGB")
    p = tmp_path / "cut.jpg"
    big.save(p, quality=95)
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])      # e.g. card removed while copying
    read_photo_info(p)                         # header is still readable
    with pytest.raises(UnreadableImageError):
        load_image_oriented(p)                 # full decode reports it


# ---------------------------------------------------------------- read-only

def _snapshot(folder: Path):
    return {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_scanning_never_modifies_or_writes_to_photo_folder(tmp_path, image_factory, monkeypatch):
    photos = tmp_path / "photos"
    for i in range(3):
        image_factory(photos / f"IMG_{i}.jpg", orientation=6, datetime_original="2026:04:12 08:00:00")
    image_factory(photos / "sub" / "x.png", fmt="PNG")
    before = _snapshot(photos)

    # Fail the test if anything tries to open a file for writing, or to
    # rename, delete or touch anything, while scanning and decoding.
    real_open = builtins.open

    def guarded_open(file, mode="r", *a, **kw):
        assert not set(mode) & set("wax+"), f"write attempt: {file} ({mode})"
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", guarded_open)
    for name in ("remove", "unlink", "rename", "replace", "utime", "chmod", "rmdir", "mkdir"):
        monkeypatch.setattr(os, name, lambda *a, _n=name, **k: pytest.fail(f"os.{_n} called"))

    for path, st in iter_photo_files(photos):
        read_photo_info(path, st)
        load_image_oriented(path)

    monkeypatch.undo()
    assert _snapshot(photos) == before       # same bytes, same modification times


def test_read_only_files_can_be_scanned(tmp_path, image_factory):
    p = image_factory(tmp_path / "locked.jpg", datetime_original="2026:04:12 08:00:00")
    os.chmod(p, stat.S_IREAD)                # sets the read-only attribute on Windows
    try:
        assert read_photo_info(p).capture_time == "2026-04-12T08:00:00"
        load_image_oriented(p)
    finally:
        os.chmod(p, stat.S_IREAD | stat.S_IWRITE)


# ---------------------------------------------------------------- Ultralytics side effect
# Root cause: importing Ultralytics replaces PIL.Image.open with a wrapper
# that raises ModuleNotFoundError('pi_heif') for unidentifiable files.
# Simulated here, so these tests also run without the AI stack. The real
# Ultralytics is tested in tests/e2e/test_ultralytics_pillow_patch.py.

def _install_fake_patch(monkeypatch):
    from utils.pil_compat import pillow_open
    genuine = pillow_open()

    def patched_open(filename, *a, **kw):
        try:
            return genuine(filename, *a, **kw)
        except Exception:
            raise ModuleNotFoundError("No module named 'pi_heif'")

    patched_open.__module__ = "ultralytics.utils.patches"
    monkeypatch.setattr(Image, "open", patched_open)


def test_patched_image_open_uses_saved_pillow_original(tmp_path, monkeypatch):
    _install_fake_patch(monkeypatch)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(UnreadableImageError, match="UnidentifiedImageError"):
        read_photo_info(bad)
    with pytest.raises(UnreadableImageError, match="UnidentifiedImageError"):
        load_image_oriented(bad)


def test_patched_image_open_without_saved_original_still_never_crashes(tmp_path, monkeypatch,
                                                                      image_factory):
    """Worst case: Image.open was replaced before it could be saved."""
    import utils.pil_compat as pc
    _install_fake_patch(monkeypatch)
    monkeypatch.setattr(pc, "_ORIGINAL_OPEN", None)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(UnreadableImageError, match="ModuleNotFoundError"):
        read_photo_info(bad)
    # good files still work through the wrapper
    good = image_factory(tmp_path / "good.jpg")
    assert read_photo_info(good).width == 400


def test_folder_scan_completes_with_patched_image_open(tmp_path, monkeypatch, image_factory):
    import utils.pil_compat as pc
    from database.database import get_or_create_event, init_db
    from scanner.folder_sync import sync_folder
    _install_fake_patch(monkeypatch)
    monkeypatch.setattr(pc, "_ORIGINAL_OPEN", None)
    photos = tmp_path / "p"
    image_factory(photos / "a.jpg")
    (photos / "b.jpg").write_bytes(b"broken")
    conn = init_db(tmp_path / "x.db")
    s = sync_folder(conn, get_or_create_event(conn, "R"), photos)
    assert (s.completed, s.new, s.unreadable) == (True, 1, 1)
