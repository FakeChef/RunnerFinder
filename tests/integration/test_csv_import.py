"""Importing real OCR diagnostic batch folders (made with stand-in detector and OCR)."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from database import csv_import
from database import database as db
from database.csv_import import ResultConflict, import_batch
from ocr.diagnostic import run_ocr_diagnostic
from tests.helpers import ColorBlobDetector, ColorTextOCR, draw_people, make_test_config

RED, BLUE = (230, 20, 20), (20, 40, 230)
CYAN, ORANGE, MAGENTA = (0, 200, 200), (250, 140, 0), (200, 0, 200)


@pytest.fixture
def photos(tmp_path):
    f = tmp_path / "photos"
    draw_people(f / "01_group.jpg", (2400, 1600),
                [(RED, (300, 200, 900, 1500)), (BLUE, (850, 200, 1400, 1500)),
                 (CYAN, (870, 600, 930, 700))])
    draw_people(f / "02_no_person.jpg", (2400, 1600), [(ORANGE, (1000, 700, 1200, 800))])
    (f / "03_corrupt.jpg").write_bytes(b"\xff\xd8 broken")
    draw_people(f / "04_sponsor_only.jpg", (2400, 1600),
                [(RED, (300, 200, 900, 1500)), (MAGENTA, (400, 500, 600, 560))])
    return f


def make_batch(tmp_path, photos, **cfg_over) -> Path:
    cfg = make_test_config(tmp_path, full_image_fallback="no_person", **cfg_over)
    engine = ColorTextOCR({CYAN: ("00123", 0.95), ORANGE: ("UT001", 0.91), MAGENTA: ("NIKE", 0.99)})
    return Path(run_ocr_diagnostic(photos, cfg, ColorBlobDetector([RED, BLUE]), engine,
                                   debug=False).run_dir)


def snapshot(folder: Path):
    return {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in folder.rglob("*") if p.is_file()}


def pairs(conn, event):
    return sorted((Path(r["file_path"]).name, r["bib_number"], r["numeric_bib"])
                  for r in conn.execute(
        "SELECT p.file_path, v.bib_number, v.numeric_bib FROM v_photo_bibs v "
        "JOIN photos p ON p.id = v.photo_id JOIN events e ON e.id = p.event_id WHERE e.name = ?",
        (event,)))


def count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_import_matches_the_batch_and_only_reads(tmp_path, photos):
    batch = make_batch(tmp_path, photos)
    before_photos, before_batch = snapshot(photos), snapshot(batch)
    conn = db.init_db(tmp_path / "idx" / "t.db")

    s = import_batch(conn, batch, "Race lite")
    assert (s.photos, s.photos_with_bibs, s.photos_failed, s.pairs) == (4, 2, 1, 2)
    assert pairs(conn, "Race lite") == [("01_group.jpg", "00123", "00123"),     # zeros kept
                                        ("02_no_person.jpg", "UT001", "001")]
    row = conn.execute("SELECT * FROM bib_detection WHERE bib_number = '00123'").fetchone()
    assert (row["readings"], row["region_ids"], row["source"], row["ocr_text"]) == (2, "C1;C2",
                                                                                   "person_body", "00123")
    ev = json.loads(row["evidence_json"])
    assert sorted(r["region_id"] for r in ev["readings"]) == ["C1", "C2"]
    assert not any(r["raw_text"] == "NIKE" for r in ev["readings"])            # rejected: CSV only
    assert count(conn, "bib_detection") == 2                                    # no rejected rows

    st = {Path(r["file_path"]).name: (r["analysis_status"], r["file_status"])
          for r in conn.execute("SELECT * FROM photos")}
    assert st["03_corrupt.jpg"] == ("error", "unreadable")
    assert st["04_sponsor_only.jpg"] == ("done", "ok")
    run = conn.execute("SELECT * FROM analysis_runs").fetchone()
    assert (run["status"], run["source"], run["source_path"]) == ("finished", "csv_import", str(batch))
    assert snapshot(photos) == before_photos and snapshot(batch) == before_batch


def test_same_batch_is_imported_once_per_event(tmp_path, photos):
    batch = make_batch(tmp_path, photos)
    conn = db.init_db(tmp_path / "idx" / "t.db")
    import_batch(conn, batch, "Race")
    again = import_batch(conn, batch, "Race")
    assert again.already_imported and count(conn, "bib_detection") == 2
    assert count(conn, "analysis_runs") == 1
    other = import_batch(conn, batch, "Race copy")                    # another event: allowed
    assert not other.already_imported and pairs(conn, "Race copy") == pairs(conn, "Race")


def test_conflicts_stop_before_anything_is_written(tmp_path, photos):
    conn = db.init_db(tmp_path / "idx" / "t.db")
    import_batch(conn, make_batch(tmp_path, photos), "Race")
    body_pairs = pairs(conn, "Race")

    torso = make_batch(tmp_path, photos, crop_strategy="torso")      # other experiment, same photos
    with pytest.raises(ResultConflict) as exc:
        import_batch(conn, torso, "Race")
    assert "already have results" in str(exc.value) and "different OCR settings" in str(exc.value)
    assert count(conn, "analysis_runs") == 1 and pairs(conn, "Race") == body_pairs

    import_batch(conn, torso, "Race torso")                            # separate event: fine
    s = import_batch(conn, torso, "Race", replace=True)               # on purpose
    assert s.warnings and count(conn, "analysis_runs") == 3
    assert conn.execute("SELECT COUNT(DISTINCT d.run_id) FROM bib_detection d JOIN photos p "
                        "ON p.id = d.photo_id JOIN events e ON e.id = p.event_id "
                        "WHERE e.name = 'Race'").fetchone()[0] == 1


def test_missing_original_is_imported_as_missing(tmp_path, photos):
    batch = make_batch(tmp_path, photos)
    (photos / "02_no_person.jpg").unlink()
    conn = db.init_db(tmp_path / "idx" / "t.db")
    s = import_batch(conn, batch, "Race")
    assert s.photo_files_missing == 1
    r = conn.execute("SELECT file_status FROM photos WHERE file_name = '02_no_person.jpg'").fetchone()
    assert r["file_status"] == "missing"
    assert ("02_no_person.jpg", "UT001", "001") in pairs(conn, "Race")    # still searchable


def test_review_overrides_become_feedback(tmp_path, photos):
    batch = make_batch(tmp_path, photos)
    own = str(batch / "photo_bibs.csv")
    with open(batch / "review_overrides.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["batch_id", "photo_identity", "bib", "action"])
        w.writerow([own, str(photos / "01_group.jpg"), "00123", "reject"])
        w.writerow([own, str(photos / "02_no_person.jpg"), "UT001", "restore"])
        w.writerow([r"G:\other\photo_bibs.csv", str(photos / "02_no_person.jpg"), "UT001", "reject"])
    conn = db.init_db(tmp_path / "idx" / "t.db")
    s = import_batch(conn, batch, "Race")
    assert (s.feedback_added, s.overrides_ignored) == (1, 1)
    fb = conn.execute("SELECT verdict, original_bib, numeric_bib FROM detection_feedback").fetchall()
    assert [tuple(r) for r in fb] == [("incorrect", "00123", "00123")]
    assert pairs(conn, "Race") == [("02_no_person.jpg", "UT001", "001")]     # rejected bib hidden


def test_import_is_all_or_nothing(tmp_path, photos, monkeypatch):
    batch = make_batch(tmp_path, photos)
    conn = db.init_db(tmp_path / "idx" / "t.db")

    def boom(*a, **k):
        raise RuntimeError("simulated failure after the photos")
    monkeypatch.setattr(csv_import, "_import_overrides", boom)
    with pytest.raises(RuntimeError):
        import_batch(conn, batch, "Race")
    assert count(conn, "photos") == 0 and count(conn, "bib_detection") == 0
    assert count(conn, "imported_batches") == 0
    assert conn.execute("SELECT status FROM analysis_runs").fetchone()[0] == "failed"
    monkeypatch.undo()
    assert import_batch(conn, batch, "Race").pairs == 2                 # retry works


def test_old_batch_without_numeric_column(tmp_path, photos):
    batch = make_batch(tmp_path, photos)
    rows = list(csv.DictReader(open(batch / "photo_bibs.csv", encoding="utf-8-sig")))
    with open(batch / "photo_bibs.csv", "w", encoding="utf-8-sig", newline="") as fh:
        fields = [f for f in rows[0] if f != "numeric_identifier"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (batch / "readings.csv").unlink()
    (batch / "report.json").unlink()
    conn = db.init_db(tmp_path / "idx" / "t.db")
    import_batch(conn, batch, "Old")
    assert pairs(conn, "Old") == [("01_group.jpg", "00123", "00123"), ("02_no_person.jpg", "UT001", "001")]
    assert json.loads(conn.execute("SELECT evidence_json FROM bib_detection LIMIT 1").fetchone()[0])[
        "readings"] is None


def test_event_name_is_required(tmp_path, photos):
    conn = db.init_db(tmp_path / "idx" / "t.db")
    with pytest.raises(ValueError):
        import_batch(conn, make_batch(tmp_path, photos), "  ")


def test_command_line_import_and_refusal(tmp_path, photos):
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[2]
    batch, dbfile = make_batch(tmp_path, photos), tmp_path / "idx" / "cli.db"

    def cli(*args):
        return subprocess.run([sys.executable, str(root / "index_cli.py"), "import-csv", str(batch),
                               "--db", str(dbfile), *args], capture_output=True, text=True, cwd=root)
    first = cli("--event", "Race")
    assert first.returncode == 0, first.stderr
    assert "Photo-bib pairs:     2" in first.stdout
    assert "already imported" in cli("--event", "Race").stdout
    other = make_batch(tmp_path / "t2", photos, crop_strategy="torso")
    batch = other
    refused = cli("--event", "Race")
    assert refused.returncode == 3 and "Refused, nothing was changed" in refused.stdout
    assert cli("--event", "Race", "--replace").returncode == 0


def test_batch_with_only_photo_bibs_csv(tmp_path, photos):
    """The UAT folders hold a copied photo_bibs.csv (and overrides) only."""
    full = make_batch(tmp_path, photos)
    uat = tmp_path / "uat" / "body_batch"
    uat.mkdir(parents=True)
    (uat / "photo_bibs.csv").write_bytes((full / "photo_bibs.csv").read_bytes())
    conn = db.init_db(tmp_path / "idx" / "t.db")
    s = import_batch(conn, uat, "UAT")
    assert (s.photos, s.photos_with_bibs, s.pairs) == (2, 2, 2) and len(s.notes) == 2
    assert pairs(conn, "UAT") == [("01_group.jpg", "00123", "00123"), ("02_no_person.jpg", "UT001", "001")]
