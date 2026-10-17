"""Per-photo saving of live OCR runs to the SQLite index (stand-in detector and OCR)."""
import csv
import json
from pathlib import Path

import pytest

from database import database as db
from database.results_store import OCRRunStore, ResultConflict, settings_hash
from ocr.diagnostic import run_ocr_diagnostic
from tests.helpers import ColorBlobDetector, ColorTextOCR, draw_people, make_test_config

RED, BLUE = (230, 20, 20), (20, 40, 230)
CYAN, ORANGE = (0, 200, 200), (250, 140, 0)


def engine():
    return ColorTextOCR({CYAN: ("00123", 0.95), ORANGE: ("UT001", 0.91)})


@pytest.fixture
def photos(tmp_path):
    f = tmp_path / "photos"
    draw_people(f / "01_group.jpg", (2400, 1600),
                [(RED, (300, 200, 900, 1500)), (BLUE, (850, 200, 1400, 1500)),
                 (CYAN, (870, 600, 930, 700))])                       # one bib, read in two crops
    draw_people(f / "02_no_person.jpg", (2400, 1600), [(ORANGE, (1000, 700, 1200, 800))])
    (f / "03_corrupt.jpg").write_bytes(b"\xff\xd8 broken")
    draw_people(f / "04_nothing.jpg", (2400, 1600), [(RED, (300, 200, 900, 1500))])
    return f


def run(tmp_path, photos, conn, event="Race", replace=False, progress=None, **cfg_over):
    cfg = make_test_config(tmp_path, full_image_fallback="no_person", **cfg_over)
    store = OCRRunStore(conn, db.get_or_create_event(conn, event), replace=replace)
    diag = run_ocr_diagnostic(photos, cfg, ColorBlobDetector([RED, BLUE]), engine(),
                              debug=False, store=store, progress=progress)
    return diag, store


def stored(conn, event="Race"):
    return sorted((Path(r["file_path"]).name, r["bib_number"], r["numeric_bib"], r["source"])
                  for r in conn.execute(
        "SELECT p.file_path, d.bib_number, d.numeric_bib, d.source FROM bib_detection d "
        "JOIN photos p ON p.id = d.photo_id JOIN events e ON e.id = p.event_id WHERE e.name = ?",
        (event,)))


def test_live_run_saves_every_photo_like_the_csv(tmp_path, photos):
    conn = db.init_db(tmp_path / "data" / "t.db")
    diag, store = run(tmp_path, photos, conn)

    assert stored(conn) == [("01_group.jpg", "00123", "00123", "person_body"),
                            ("02_no_person.jpg", "UT001", "001", "full_image")]
    with open(Path(diag.run_dir) / "photo_bibs.csv", encoding="utf-8-sig") as fh:
        csv_pairs = sorted((Path(r["file"]).name, r["bib"]) for r in csv.DictReader(fh))
    assert csv_pairs == [(f, b) for f, b, _, _ in stored(conn)]          # DB == CSV

    status = {Path(r["file_path"]).name: (r["analysis_status"], r["file_status"], r["run_id"])
              for r in conn.execute("SELECT * FROM photos")}
    assert status == {"01_group.jpg": ("done", "ok", store.run_id),
                      "02_no_person.jpg": ("done", "ok", store.run_id),
                      "03_corrupt.jpg": ("error", "unreadable", None),
                      "04_nothing.jpg": ("done", "ok", store.run_id)}

    ev = json.loads(conn.execute("SELECT evidence_json FROM bib_detection WHERE bib_number = '00123'")
                    .fetchone()[0])
    assert sorted(r["region_id"] for r in ev["readings"]) == ["C1", "C2"]
    assert all(r["raw_text"] == "00123" for r in ev["readings"])        # accepted readings only

    r = conn.execute("SELECT * FROM analysis_runs WHERE id = ?", (store.run_id,)).fetchone()
    assert (r["status"], r["photos_processed"], r["source"], r["device"]) == ("finished", 4, "ocr", None)
    assert r["settings_hash"] == settings_hash(diag.settings)


def test_stopped_run_keeps_finished_photos(tmp_path, photos):
    conn = db.init_db(tmp_path / "data" / "t.db")

    def stop_at_third(i, n, path):
        if i == 3:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, photos, conn, progress=stop_at_third)
    done = [Path(r["file_path"]).name for r in conn.execute(
        "SELECT file_path FROM photos WHERE analysis_status = 'done' ORDER BY file_path")]
    assert done == ["01_group.jpg", "02_no_person.jpg"]
    r = conn.execute("SELECT status, photos_processed FROM analysis_runs").fetchone()
    assert tuple(r) == ("cancelled", 2)


def test_second_run_is_refused_unless_replace(tmp_path, photos):
    conn = db.init_db(tmp_path / "data" / "t.db")
    _, first = run(tmp_path, photos, conn)
    pid = conn.execute("SELECT photo_id FROM bib_detection WHERE bib_number = '00123'").fetchone()[0]
    db.add_feedback(conn, pid, "incorrect", original_bib="00123")
    conn.commit()
    before = stored(conn)

    with pytest.raises(ResultConflict, match="already have results"):
        run(tmp_path, photos, conn)
    assert stored(conn) == before
    assert conn.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 1   # no run started

    _, second = run(tmp_path, photos, conn, replace=True)
    assert second.warnings and stored(conn) == before                    # replaced, no duplicates
    assert conn.execute("SELECT COUNT(DISTINCT run_id) FROM bib_detection").fetchone()[0] == 1
    assert conn.execute("SELECT run_id FROM bib_detection LIMIT 1").fetchone()[0] == second.run_id
    # the manual verdict survived the re-analysis and still hides the bib
    assert conn.execute("SELECT COUNT(*) FROM detection_feedback").fetchone()[0] == 1
    assert [r["bib_number"] for r in conn.execute("SELECT bib_number FROM v_photo_bibs")] == ["UT001"]


def test_different_settings_are_not_mixed_into_one_event(tmp_path, photos):
    conn = db.init_db(tmp_path / "data" / "t.db")
    run(tmp_path, photos, conn)
    other = tmp_path / "other"
    draw_people(other / "05.jpg", (2400, 1600), [(ORANGE, (1000, 700, 1200, 800))])
    with pytest.raises(ResultConflict, match="different OCR settings"):
        run(tmp_path, other, conn, crop_strategy="torso")
    run(tmp_path, other, conn, event="Race torso", crop_strategy="torso")    # separate event: fine
    assert stored(conn, "Race torso") == [("05.jpg", "UT001", "001", "full_image")]


def test_failed_reanalysis_keeps_previous_results(tmp_path):
    conn = db.init_db(tmp_path / "t.db")
    ev = db.get_or_create_event(conn, "E")
    pid = db.insert_photo(conn, ev, {"file_path": "x", "path_key": "x", "file_name": "x"})
    run_id = db.start_run(conn, ev, source="ocr", source_folder="f")
    db.save_photo_result(conn, pid, run_id, [{"bib_number": "123", "numeric_bib": "123",
                                              "source": "person_body"}])
    db.save_photo_result(conn, pid, run_id, [], error="RuntimeError: GPU out of memory")
    assert conn.execute("SELECT bib_number FROM bib_detection").fetchone()[0] == "123"
    assert conn.execute("SELECT analysis_status FROM photos").fetchone()[0] == "error"


def test_one_photo_is_all_or_nothing(tmp_path):
    conn = db.init_db(tmp_path / "t.db")
    ev = db.get_or_create_event(conn, "E")
    pid = db.insert_photo(conn, ev, {"file_path": "x", "path_key": "x", "file_name": "x"})
    run_id = db.start_run(conn, ev, source="ocr", source_folder="f")
    db.save_photo_result(conn, pid, run_id, [{"bib_number": "123", "source": "person_body"}])
    bad = [{"bib_number": "456", "source": "person_body"}, {"bib_number": "", "source": "x"}]
    with pytest.raises(Exception):
        db.save_photo_result(conn, pid, run_id, bad)                  # 2nd row breaks a CHECK
    assert [r[0] for r in conn.execute("SELECT bib_number FROM bib_detection")] == ["123"]


def test_settings_hash_ignores_device_and_sample_but_not_results_settings():
    base = {"processing_mode": "lite1", "crop_strategy": "body", "crop_fractions": [0.1, 0.9, 0.15],
            "detector": "yolo26n.pt (conf 0.25, imgsz 1280, device cpu)",
            "ocr_engine": "PaddleOCR a + b (device cpu, mkldnn False)", "limit": 50, "debug": True}
    gpu = {**base, "ocr_engine": "PaddleOCR a + b (device gpu:0, mkldnn False)",
           "detector": "yolo26n.pt (conf 0.25, imgsz 1280, device cuda:0)", "limit": 370, "debug": False}
    assert settings_hash(base) == settings_hash(gpu)
    assert settings_hash(base) == settings_hash({**base, "crop_fractions": (0.1, 0.9, 0.15)})
    assert settings_hash(base) != settings_hash({**base, "crop_strategy": "torso"})
    assert settings_hash(base) != settings_hash({**base, "ocr_engine": "PaddleOCR a + b (device cpu, mkldnn True)"})
