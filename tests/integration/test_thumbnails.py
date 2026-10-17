"""Thumbnail cache (V0.3.1): read-only originals, correct orientation, rebuild, write location."""
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from database import database as db
from indexing.analyze import analyze_folder
from ocr import diagnostic
from scanner.photo_scanner import compute_fingerprint, load_image_oriented
from tests.helpers import ColorBlobDetector, ColorTextOCR, draw_people, make_test_config
from utils import thumbnails
from utils.thumbnails import (display_path, ensure_thumbnail, existing_thumbnail, save_from_image,
                              thumbnail_path)

RED, CYAN = (230, 20, 20), (0, 200, 200)


def fp(path):
    return compute_fingerprint(path, Path(path).stat().st_size)


def state(folder):
    return {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in Path(folder).rglob("*") if p.is_file()}


def all_files(root):
    return {p for p in Path(root).rglob("*") if p.is_file()}


def colored_quadrant(img):
    """Which quadrant (row, col) holds the red quarter of make_image()."""
    img = img.convert("RGB")
    w, h = img.size
    best = None
    for r in (0, 1):
        for c in (0, 1):
            px = img.getpixel((int((c + 0.5) * w / 2), int((r + 0.5) * h / 2)))
            if px[0] > 150 and px[1] < 100:
                best = (r, c)
    return best


def test_thumbnail_is_made_correctly_and_originals_stay_untouched(tmp_path, image_factory):
    photos = tmp_path / "photos"
    p = image_factory(photos / "a.jpg", size=(3000, 2000))
    cache = tmp_path / "data" / "thumbnails"
    before = state(photos)

    t = ensure_thumbnail(p, cache, fp(p))
    assert t == thumbnail_path(cache, fp(p)) and t.parent.name == fp(p)[:2]
    with Image.open(t) as im:
        assert im.format == "JPEG" and max(im.size) == 512 and im.size == (512, 341)
        assert im.getexif().get(0x0112) is None                  # no EXIF orientation inside
    assert state(photos) == before                              # bytes and mtime unchanged


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_is_applied(tmp_path, image_factory, orientation):
    p = image_factory(tmp_path / "photos" / f"o{orientation}.jpg", size=(400, 200), orientation=orientation)
    expected = load_image_oriented(p)                           # how the app/OCR see the photo
    t = ensure_thumbnail(p, tmp_path / "thumbs", fp(p))
    with Image.open(t) as im:
        assert (im.width > im.height) == (expected.width > expected.height)
        assert colored_quadrant(im) == colored_quadrant(expected)
    # the same result when made from the already decoded image (analyze path)
    t2 = save_from_image(expected, tmp_path / "thumbs2", fp(p))
    with Image.open(t2) as im:
        assert colored_quadrant(im) == colored_quadrant(expected)


def test_deleted_thumbnail_is_rebuilt(tmp_path, image_factory):
    p = image_factory(tmp_path / "photos" / "a.jpg")
    cache = tmp_path / "thumbs"
    t = ensure_thumbnail(p, cache, fp(p))
    t.unlink()
    assert existing_thumbnail(cache, fp(p)) is None
    assert ensure_thumbnail(p, cache, fp(p)) == t and t.is_file()


def test_nothing_is_written_outside_the_thumbnail_cache(tmp_path, image_factory):
    p = image_factory(tmp_path / "photos" / "a.jpg")
    cache = tmp_path / "data" / "thumbnails"
    before = all_files(tmp_path)
    ensure_thumbnail(p, cache, fp(p))
    save_from_image(load_image_oriented(p), cache, "ab" * 32)
    new = all_files(tmp_path) - before
    assert new and all(f.resolve().is_relative_to(cache.resolve()) for f in new)
    assert not any(f.name.startswith(".tmp_") for f in new)    # no temporary files left
    with pytest.raises(ValueError):
        thumbnail_path(cache, "../../evil")
    assert save_from_image(load_image_oriented(p), cache, "../../evil") is None


def test_fallbacks_never_break(tmp_path, image_factory):
    cache = tmp_path / "thumbs"
    missing = tmp_path / "photos" / "gone.jpg"
    assert display_path(missing, cache, "cd" * 32) == missing          # original path as fallback
    bad = tmp_path / "photos" / "bad.jpg"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"\xff\xd8 not a jpeg")
    assert ensure_thumbnail(bad, cache, fp(bad)) is None
    assert display_path(bad, cache, None) == bad                       # no fingerprint: original


def test_analyze_saves_thumbnails_without_a_second_decode(tmp_path, monkeypatch):
    folder = tmp_path / "photos"
    for i in range(4):
        draw_people(folder / f"r{i}.jpg", (1200, 800), [(RED, (200, 100, 500, 700)), (CYAN, (300, 300, 360, 340))])
    cfg = make_test_config(tmp_path, processing_mode="lite1")
    conn = db.init_db(cfg.paths.database_file)
    before = state(folder)

    decodes = []
    real = diagnostic.load_image_with_orientation
    monkeypatch.setattr(diagnostic, "load_image_with_orientation",
                        lambda path: decodes.append(path) or real(path))

    def no_lazy_open(*a, **k):
        raise AssertionError("the original was opened again for the thumbnail")
    monkeypatch.setattr(thumbnails, "pillow_open", lambda: no_lazy_open)

    s = analyze_folder(conn, folder, "Race", cfg,
                       lambda: (ColorBlobDetector([RED]), ColorTextOCR({CYAN: ("00123", 0.95)})))
    assert s.processed == 4 and s.thumbnails == 4
    assert len(decodes) == 4                                           # one decode per photo
    for r in conn.execute("SELECT fingerprint FROM photos"):
        assert existing_thumbnail(cfg.paths.thumbnail_cache, r["fingerprint"])
    assert state(folder) == before
    for f in all_files(tmp_path) - set(state(folder)):
        rel = f.resolve()
        assert rel.is_relative_to(cfg.paths.data_dir.resolve())        # DB, CSVs, thumbnails only


def test_thumbnails_can_be_switched_off(tmp_path):
    folder = tmp_path / "photos"
    draw_people(folder / "r.jpg", (1200, 800), [(RED, (200, 100, 500, 700))])
    cfg = make_test_config(tmp_path, processing_mode="lite1")
    cfg.thumbnails.generate_during_analyze = False
    conn = db.init_db(cfg.paths.database_file)
    s = analyze_folder(conn, folder, "Race", cfg,
                       lambda: (ColorBlobDetector([RED]), ColorTextOCR({})))
    assert s.processed == 1 and s.thumbnails == 0
    assert not Path(cfg.paths.thumbnail_cache).exists()
