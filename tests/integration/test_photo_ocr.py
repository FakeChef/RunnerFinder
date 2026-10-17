"""
Per-photo OCR pipeline on real image files, with colour stand-ins for YOLO
and PaddleOCR so that every position is known exactly.
"""
import sqlite3

import pytest

from config import AppConfig
from database.database import get_or_create_event, init_db
from detection.bib_detector import RegionFinder
from detection.geometry import Box
from ocr.photo_ocr import (BIB_READ, NO_TEXT, OCR_ERROR, TEXT_NO_VALID_BIB, TOO_SMALL,
                           PhotoOCR, detection_rows)
from scanner.photo_scanner import load_image_with_orientation
from tests.helpers import ColorBlobDetector, ColorTextOCR, draw_people

RED, BLUE = (230, 20, 20), (20, 40, 230)              # runners
CYAN, YELLOW, MAGENTA, ORANGE = (0, 200, 200), (240, 230, 20), (200, 0, 200), (250, 140, 0)


def run(tmp_path, people, texts, size=(2400, 1600), **cfg_changes):
    p = draw_people(tmp_path / "photo.png", size, people, fmt="PNG")
    image, _ = load_image_with_orientation(p)
    cfg = AppConfig()
    for key, value in cfg_changes.items():
        section, name = key.split("__")
        setattr(getattr(cfg, section), name, value)
    engine = ColorTextOCR(texts)
    pipeline = PhotoOCR(cfg, RegionFinder(cfg.detection, ColorBlobDetector([RED, BLUE])), engine)
    return pipeline.process(image), engine


# Runner A (red) partly hidden behind runner B (blue); B's bib (cyan) lies
# inside BOTH expanded body crops.
OVERLAP = [(RED, (300, 200, 900, 1500)), (BLUE, (850, 200, 1400, 1500)), (CYAN, (870, 600, 930, 700))]


def test_overlapping_crops_give_one_bib_per_photo_with_all_evidence(tmp_path):
    res, _ = run(tmp_path, OVERLAP, {CYAN: ("10258", 0.95)})
    assert [b.bib for b in res.bibs] == ["10258"]
    bib = res.bibs[0]
    assert bib.readings == 2 and bib.region_ids == ("C1", "C2")
    raw = [r for r in res.readings if r.accepted]
    assert len(raw) == 2 and {r.region_id for r in raw} == {"C1", "C2"}
    # both readings point at the same place in the ORIGINAL photo
    assert all(r.line_box == Box(870, 600, 930, 700) for r in raw)


def test_bib_is_not_assigned_to_the_person_whose_crop_it_was_read_in(tmp_path):
    res, _ = run(tmp_path, OVERLAP, {CYAN: ("10258", 0.95)})
    # The bib is physically on runner B (P2), but was also read in A's crop.
    from_a = next(r for r in res.readings if r.region_id == "C1")
    assert from_a.region_person_index == 1            # evidence: where it was read
    assert not hasattr(res.bibs[0], "person_index")   # result: photo-level only


def test_ocr_receives_full_resolution_crops(tmp_path):
    res, engine = run(tmp_path, OVERLAP, {CYAN: ("10258", 0.95)})
    assert engine.calls == [(r.region.box.width, r.region.box.height) for r in res.regions]
    assert res.detection.detection_size == (1280, 853)        # detection used a smaller copy


def test_region_statuses_explain_why_no_bib_was_read(tmp_path):
    people = [(RED, (200, 200, 600, 1500)), (BLUE, (1200, 200, 1600, 1500)),
              (MAGENTA, (300, 500, 450, 560))]               # sponsor word on runner A
    res, _ = run(tmp_path, people, {MAGENTA: ("NIKE", 0.99)}, detection__full_image_fallback="never")
    statuses = {r.region_id: r.status for r in res.regions}
    assert statuses == {"C1": TEXT_NO_VALID_BIB, "C2": NO_TEXT}
    assert res.bibs == [] and res.status.startswith("no_valid_bib_read")
    assert next(r for r in res.readings if r.normalized == "NIKE").reason == "no_digit"


def test_ocr_error_on_one_crop_does_not_stop_the_photo(tmp_path):
    people = [(RED, (200, 200, 600, 1500)), (BLUE, (1200, 200, 1700, 1500)),   # different widths
              (CYAN, (1300, 500, 1450, 600))]
    p = draw_people(tmp_path / "photo.png", (2400, 1600), people, fmt="PNG")
    image, _ = load_image_with_orientation(p)
    cfg = AppConfig()
    finder = RegionFinder(cfg.detection, ColorBlobDetector([RED, BLUE]))
    first = finder.find(image).regions[0].box
    engine = ColorTextOCR({CYAN: ("10311", 0.9)}, fail_on_sizes={(first.width, first.height)})
    res = PhotoOCR(cfg, finder, engine).process(image)
    statuses = {r.region_id: r.status for r in res.regions}
    assert statuses == {"C1": OCR_ERROR, "C2": BIB_READ}
    assert "simulated OCR failure" in res.regions[0].error
    assert [b.bib for b in res.bibs] == ["10311"]


def test_crop_too_small_is_reported_not_silently_dropped(tmp_path):
    res, engine = run(tmp_path, OVERLAP, {CYAN: ("10258", 0.95)}, ocr__min_crop_side_px=100000,
                      detection__full_image_fallback="never")
    assert {r.status for r in res.regions} == {TOO_SMALL} and engine.calls == []


def test_low_confidence_reading_is_kept_as_evidence_but_rejected(tmp_path):
    res, _ = run(tmp_path, OVERLAP, {CYAN: ("10258", 0.30)}, detection__full_image_fallback="never")
    assert res.bibs == []
    assert {r.reason for r in res.readings} == {"low_ocr_confidence"}


def test_fallback_after_crops_finds_bib_below_crop_and_ignores_banner(tmp_path):
    people = [(RED, (300, 200, 900, 1500)),
              (YELLOW, (500, 1420, 700, 1480)),               # bib low on the leg: below the crop
              (MAGENTA, (1800, 100, 2200, 200))]              # banner, outside every person
    res, _ = run(tmp_path, people, {YELLOW: ("10258", 0.9), MAGENTA: ("2026", 0.95)})
    assert res.regions[0].status == NO_TEXT                   # the body crop (10-90 %) misses it
    assert res.fallback_used == "no_valid_bib_in_crops"
    assert [b.bib for b in res.bibs] == ["10258"]
    banner = next(r for r in res.readings if r.bib == "2026")
    assert banner.reason == "full_image_outside_person" and banner.region_source == "full_image"


def test_fallback_needs_higher_confidence(tmp_path):
    people = [(RED, (300, 200, 900, 1500)), (YELLOW, (500, 1420, 700, 1480))]
    res, _ = run(tmp_path, people, {YELLOW: ("10258", 0.70)})
    assert res.bibs == []
    assert next(r for r in res.readings if r.bib == "10258").reason == "full_image_low_confidence"


def test_no_person_uses_whole_photo(tmp_path):
    res, _ = run(tmp_path, [(ORANGE, (1000, 700, 1200, 800))], {ORANGE: ("UT001", 0.92)})
    assert res.fallback_used == "no_person" and [b.bib for b in res.bibs] == ["UT001"]


def test_registered_list_rejects_unknown_numbers(tmp_path):
    p = draw_people(tmp_path / "photo.png", (2400, 1600), OVERLAP, fmt="PNG")
    image, _ = load_image_with_orientation(p)
    cfg = AppConfig()
    cfg.detection.full_image_fallback = "never"
    pipe = PhotoOCR(cfg, RegionFinder(cfg.detection, ColorBlobDetector([RED, BLUE])),
                    ColorTextOCR({CYAN: ("10258", 0.95)}), registered={"10311"})
    res = pipe.process(image)
    assert res.bibs == [] and {r.reason for r in res.readings} == {"not_registered"}


def test_leading_zeros_survive_the_pipeline(tmp_path):
    res, _ = run(tmp_path, OVERLAP, {CYAN: ("00123", 0.95)})
    assert [b.bib for b in res.bibs] == ["00123"]


def test_duplicate_readings_give_one_search_result_in_database(tmp_path):
    res, _ = run(tmp_path, OVERLAP + [(ORANGE, (1000, 1000, 1100, 1060))],
                 {CYAN: ("10258", 0.95), ORANGE: ("UT001", 0.9)})
    rows = detection_rows(res)
    assert sum(r["bib_number"] == "10258" for r in rows) == 2        # evidence kept
    conn = init_db(tmp_path / "t.db")
    event = get_or_create_event(conn, "Race")
    photo = conn.execute("INSERT INTO photos (event_id, file_path, path_key, file_name) "
                         "VALUES (?, 'p', 'p', 'p')", (event,)).lastrowid
    cols = ["photo_id", *rows[0].keys()]
    conn.executemany(f"INSERT INTO bib_detection ({', '.join(cols)}) VALUES "
                     f"({', '.join('?' * len(cols))})", [[photo, *r.values()] for r in rows])
    found = conn.execute("SELECT bib_number FROM v_photo_bibs ORDER BY bib_number").fetchall()
    assert [r[0] for r in found] == ["10258", "UT001"]               # one result per bib
    assert conn.execute("SELECT COUNT(*) FROM bib_detection WHERE person_bbox_x IS NOT NULL"
                        ).fetchone()[0] == 0                         # no person assignment
