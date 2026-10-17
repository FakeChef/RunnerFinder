"""OCR diagnostic run end to end with stand-in detector and OCR engine."""
import builtins
import csv
import hashlib
from pathlib import Path

import pytest

from ocr.diagnostic import REVIEW_HEADERS, run_ocr_diagnostic, summarize
from tests.helpers import ColorBlobDetector, ColorTextOCR, draw_people, make_test_config
from utils.xlsx_writer import read_xlsx_text

RED, BLUE = (230, 20, 20), (20, 40, 230)
CYAN, ORANGE, MAGENTA = (0, 200, 200), (250, 140, 0), (200, 0, 200)


@pytest.fixture
def photos(tmp_path):
    f = tmp_path / "photos"
    draw_people(f / "01_group.jpg", (2400, 1600),
                [(RED, (300, 200, 900, 1500)), (BLUE, (850, 200, 1400, 1500)),
                 (CYAN, (870, 600, 930, 700))])                       # one bib, two crops
    draw_people(f / "02_no_person.jpg", (2400, 1600), [(ORANGE, (1000, 700, 1200, 800))])
    (f / "03_corrupt.jpg").write_bytes(b"\xff\xd8 broken")
    draw_people(f / "04_sponsor_only.jpg", (2400, 1600),
                [(RED, (300, 200, 900, 1500)), (MAGENTA, (400, 500, 600, 560))])
    return f


def engine():
    return ColorTextOCR({CYAN: ("00123", 0.95), ORANGE: ("UT001", 0.91), MAGENTA: ("NIKE", 0.99)})


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_ocr_diagnostic_outputs(tmp_path, photos):
    cfg = make_test_config(tmp_path)
    cfg.detection.full_image_fallback = "no_person"
    diag = run_ocr_diagnostic(photos, cfg, ColorBlobDetector([RED, BLUE]), engine())
    s = summarize(diag)
    assert (s["photos_processed"], s["photos_failed"]) == (3, 1)
    assert s["photos_with_bib_read"] == 2 and s["photo_bib_pairs_deduplicated"] == 2
    assert s["duplicate_readings_merged"] == 1
    assert s["rejections_by_reason"].get("no_digit") == 1

    run = Path(diag.run_dir)
    for name in ("report.md", "report.json", "photos.csv", "photo_bibs.csv", "readings.csv",
                 "regions.csv", "ground_truth_review.xlsx", "ground_truth_review.csv",
                 "REVIEW_INSTRUCTIONS.txt"):
        assert (run / name).exists(), name

    bibs = read_csv(run / "photo_bibs.csv")
    assert [(Path(b["file"]).name, b["bib"], b["readings"], b["region_ids"]) for b in bibs] == [
        ("01_group.jpg", "00123", "2", "C1;C2"), ("02_no_person.jpg", "UT001", "1", "T1")]
    assert {"review_verdict", "corrected_bib"} <= set(bibs[0])

    readings = read_csv(run / "readings.csv")
    assert {r["accepted"] for r in readings} == {"True", "False"}
    nike = next(r for r in readings if r["normalized"] == "NIKE")
    assert nike["rejection_reason"] == "no_digit" and nike["raw_text"] == "NIKE"
    assert all(r["region_id"] and r["line_w"] for r in readings)

    regions = read_csv(run / "regions.csv")
    assert {r["status"] for r in regions} >= {"bib_read", "text_but_no_valid_bib"}

    review = read_xlsx_text(run / "ground_truth_review.xlsx")
    assert review[0] == REVIEW_HEADERS and len(review) == 5          # header + 4 photos
    assert review[1][3] == "00123"                                   # leading zeros kept
    assert review[3][5].startswith("FAILED")                          # corrupt photo listed
    assert all(row[6] == "" for row in review[1:])                    # true_bibs left for the human

    md = (run / "report.md").read_text(encoding="utf-8")
    assert "does NOT measure OCR accuracy or search recall" in md
    assert "Per OCR call, person crop" in md
    ok = [r for r in diag.rows if r.status == "ok"]
    assert all(Path(r.overview_image).exists() for r in ok)


def test_ocr_diagnostic_never_writes_outside_data_and_keeps_originals(tmp_path, photos, monkeypatch):
    cfg = make_test_config(tmp_path)
    before = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
              for p in photos.rglob("*") if p.is_file()}
    root = cfg.paths.data_dir.resolve()
    real_open = builtins.open

    def guarded(file, mode="r", *a, **k):
        if set(mode) & set("wax+"):
            assert Path(file).resolve().is_relative_to(root), f"write outside data: {file}"
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded)
    run_ocr_diagnostic(photos, cfg, ColorBlobDetector([RED, BLUE]), engine())
    monkeypatch.undo()
    after = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
             for p in photos.rglob("*") if p.is_file()}
    assert after == before
