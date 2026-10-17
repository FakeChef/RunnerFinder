"""filter-bibs (V0.3.1): dry run by default, apply with backup, nothing deleted, clear = rollback."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from database import database as db
from indexing.filter_bibs import rescore_event
from ocr.bib_filter import BibCandidateFilter
from search.db_search import search

ROOT = Path(__file__).resolve().parents[2]

# (bib, numeric, confidence, raw OCR text) as a V0.2 run stored them (no filter result yet)
ROWS = [("B60914", "60914", 0.97, "B 60914"), ("13655", "13655", 0.96, "13-655"),
        ("2026", "2026", 1.0, "2026"), ("10KM", "10", 0.9, "10KM"), ("68564", "68564", 0.8, "68564")]


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "data" / "idx.db"
    conn = db.init_db(path)
    ev = db.get_or_create_event(conn, "Race")
    run = db.start_run(conn, ev, source="ocr", source_folder="f")
    for i, (bib, num, conf, raw) in enumerate(ROWS):
        pid = db.insert_photo(conn, ev, {"file_path": f"p{i}.jpg", "path_key": f"p{i}.jpg", "file_name": f"p{i}.jpg"})
        db.save_photo_result(conn, pid, run, [{
            "bib_number": bib, "numeric_bib": num, "confidence": conf, "ocr_text": raw,
            "source": "person_body", "evidence_json": json.dumps({"readings": [{"raw_text": raw}]})}])
    yield conn, path
    conn.close()


def table_hash(conn):
    rows = conn.execute("SELECT * FROM bib_detection ORDER BY id").fetchall()
    return len(rows), hashlib.sha256(repr([tuple(r) for r in rows]).encode()).hexdigest()


def visible(conn, q, **kw):
    return [c.bib for g in search(conn, q, **kw)[0] for c in g.candidates]


def test_dry_run_changes_nothing(index):
    conn, path = index
    before = table_hash(conn)
    s = rescore_event(conn, path, "Race", BibCandidateFilter())
    assert not s.applied and not s.backup and table_hash(conn) == before
    # V1.3.1: the lone "2026" is uncertain (searchable, low confidence), bag-check and 10KM are noise
    assert (s.rows, s.visible_before, s.visible_after, s.newly_hidden) == (5, 5, 3, 2)
    assert s.reasons["likely_storage_number"] == 1 and s.reasons["possible_year"] == 1
    assert dict(s.status_after) == {"accepted": 2, "uncertain": 1, "noise": 2}
    assert not (path.parent / "backups").exists()


def test_apply_hides_noise_keeps_every_row_and_makes_a_backup(index):
    conn, path = index
    n_before, _ = table_hash(conn)
    s = rescore_event(conn, path, "Race", BibCandidateFilter(), apply=True)
    assert s.applied and Path(s.backup).is_file()
    assert table_hash(conn)[0] == n_before                                  # nothing deleted
    assert visible(conn, "60914") == ["B60914"] and visible(conn, "68564") == ["68564"]
    assert visible(conn, "13655") == [] and visible(conn, "10") == []
    assert [g.status for g in search(conn, "2026")[0]] == ["uncertain"]      # shown, low confidence
    _, hidden = search(conn, "13655")
    assert [(c.bib, c.noise_filtered, c.filter_reasons) for c in hidden] == [
        ("13655", True, "likely_storage_number")]
    shown = search(conn, "13655", include_noise=True)[0][0].candidates[0]
    assert shown.noise_filtered and "likely_storage_number" in shown.filter_reasons
    ev = json.loads(conn.execute("SELECT evidence_json FROM bib_detection WHERE bib_number='13655'").fetchone()[0])
    assert ev["readings"] == [{"raw_text": "13-655"}]                       # other evidence kept
    assert ev["filter"]["accepted"] is False and ev["filter"]["rules"] == BibCandidateFilter().rules_hash


def test_clear_is_a_full_rollback(index):
    conn, path = index
    rescore_event(conn, path, "Race", BibCandidateFilter(), apply=True)
    s = rescore_event(conn, path, "Race", BibCandidateFilter(), apply=True, clear=True)
    assert s.visible_after == 5 and s.newly_visible == 2
    assert visible(conn, "13655") == ["13655"] and visible(conn, "2026") == ["2026"]
    assert all("filter" not in json.loads(r[0]) for r in conn.execute("SELECT evidence_json FROM bib_detection"))


def test_photographer_verdict_beats_the_filter(index):
    conn, path = index
    rescore_event(conn, path, "Race", BibCandidateFilter(), apply=True)
    pid = conn.execute("SELECT photo_id FROM bib_detection WHERE bib_number='13655'").fetchone()[0]
    db.add_feedback(conn, pid, "correct", original_bib="13655")
    conn.commit()
    assert visible(conn, "13655") == ["13655"]


def test_rows_without_filter_result_stay_visible(index):
    conn, _ = index
    assert visible(conn, "2026") == ["2026"]            # V0.2 data before filter-bibs: unchanged behaviour


def test_cli_is_a_dry_run_unless_apply(index):
    conn, path = index
    before = table_hash(conn)

    def cli(*extra):
        return subprocess.run([sys.executable, str(ROOT / "index_cli.py"), "filter-bibs", "--event", "Race",
                               "--db", str(path), *extra], capture_output=True, text=True, cwd=ROOT,
                              encoding="utf-8", errors="replace")
    r = cli()
    assert r.returncode == 0 and "DRY RUN" in r.stdout and "likely_storage_number" in r.stdout
    assert table_hash(conn) == before
    r = cli("--apply")
    assert r.returncode == 0 and "APPLIED" in r.stdout and "Backup before the change" in r.stdout
    assert table_hash(conn) != before and table_hash(conn)[0] == before[0]
    unknown = subprocess.run([sys.executable, str(ROOT / "index_cli.py"), "filter-bibs", "--event", "Nope",
                              "--db", str(path)], capture_output=True, text=True, cwd=ROOT,
                             encoding="utf-8", errors="replace")
    assert unknown.returncode == 2 and "no event named" in unknown.stdout


def test_new_results_get_the_filter_when_stored(tmp_path):
    """A live analysis (photo_bib_rows via OCRRunStore) attaches the filter result directly."""
    from tests.integration.test_results_store import engine
    from database.results_store import OCRRunStore
    from ocr.diagnostic import run_ocr_diagnostic
    from tests.helpers import ColorBlobDetector, draw_people, make_test_config
    RED, CYAN = (230, 20, 20), (0, 200, 200)
    f = tmp_path / "photos"
    draw_people(f / "a.jpg", (2400, 1600), [(RED, (300, 200, 900, 1500)), (CYAN, (870, 600, 930, 700))])
    cfg = make_test_config(tmp_path, full_image_fallback="no_person")
    conn = db.init_db(cfg.paths.database_file)
    store = OCRRunStore(conn, db.get_or_create_event(conn, "Race"))
    run_ocr_diagnostic(f, cfg, ColorBlobDetector([RED]), engine(), debug=False, store=store)
    ev = json.loads(conn.execute("SELECT evidence_json FROM bib_detection").fetchone()[0])
    assert ev["filter"]["accepted"] is True and ev["readings"]                 # "00123": 5 digits
