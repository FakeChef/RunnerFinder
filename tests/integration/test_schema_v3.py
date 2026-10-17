"""Schema 3: new databases, and the upgrade of real schema-2 database files."""
import sqlite3
from pathlib import Path

import pytest

from database import migrations
from database.database import SCHEMA_VERSION, SchemaError, init_db

V2_SCHEMA = Path(__file__).resolve().parents[1] / "fixtures" / "schema_v2.sql"


def make_v2_db(path: Path) -> Path:
    """A schema-2 database file with one photo, AI rows and feedback, as Phase 3-5 left it."""
    c = sqlite3.connect(path)
    c.executescript(V2_SCHEMA.read_text(encoding="utf-8"))
    c.executescript("""
        PRAGMA user_version = 2;
        INSERT INTO events (id, name) VALUES (1, 'Race');
        INSERT INTO photos (id, event_id, file_path, path_key, file_name, analysis_status)
            VALUES (1, 1, 'D:/p/a.jpg', 'd:/p/a.jpg', 'a.jpg', 'done');
        INSERT INTO bib_detection (id, photo_id, bib_number, confidence, source, ocr_text)
            VALUES (7, 1, 'A18866', 0.9, 'person_body', 'A18866'),
                   (8, 1, '00123', 0.8, 'full_image', '00123'),
                   (9, 1, NULL, NULL, 'person_body', 'no_text_detected');
        INSERT INTO detection_feedback (photo_id, detection_id, verdict, original_bib)
            VALUES (1, 8, 'incorrect', '00123');
        INSERT INTO detection_feedback (photo_id, verdict, corrected_bib)
            VALUES (1, 'added', 'UT001');
    """)
    c.commit()
    c.close()
    return path


def columns(conn, table):
    return [(r[1], r[2], r[3], r[5]) for r in conn.execute(f"PRAGMA table_info({table})")]


def objects(conn):
    return {(r[0], r[1]) for r in conn.execute(
        "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")}


def test_new_database_is_schema_3(tmp_path):
    c = init_db(tmp_path / "new.db")
    assert SCHEMA_VERSION == 3
    assert c.execute("PRAGMA user_version").fetchone()[0] == 3
    names = {name for _, name in objects(c)}
    assert {"imported_batches", "ix_detection_numeric", "ix_feedback_numeric", "v_photo_bibs"} <= names
    det = [col[0] for col in columns(c, "bib_detection")]
    assert {"numeric_bib", "readings", "region_ids", "letter_digit_fix", "evidence_json",
            "ocr_text", "det_confidence", "person_bbox_x"} <= set(det)      # evidence fields kept
    assert {"analysis_settings_hash", "analysis_device"} <= {col[0] for col in columns(c, "photos")}


def test_v2_upgrade_keeps_data_and_matches_new_schema(tmp_path):
    db = make_v2_db(tmp_path / "old.db")
    c = init_db(db)
    assert c.execute("PRAGMA user_version").fetchone()[0] == 3
    assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    rows = c.execute("SELECT id, bib_number, numeric_bib, ocr_text FROM bib_detection ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [(7, "A18866", "18866", "A18866"), (8, "00123", "00123", "00123"),
                                        (9, None, None, "no_text_detected")]
    fb = c.execute("SELECT verdict, detection_id, numeric_bib FROM detection_feedback ORDER BY id").fetchall()
    assert [tuple(r) for r in fb] == [("incorrect", 8, "00123"), ("added", None, "001")]
    view = sorted(tuple(r) for r in c.execute(
        "SELECT bib_number, numeric_bib, verification FROM v_photo_bibs"))
    # 00123 stays hidden by the 'incorrect' verdict; leading zeros are kept
    assert view == [("A18866", "18866", "unverified"), ("UT001", "001", "added")]

    # Same structure as a database created new with schema 3.
    fresh = init_db(tmp_path / "fresh.db")
    for table in ("events", "event_bibs", "analysis_runs", "photos", "bib_detection",
                  "detection_feedback", "imported_batches"):
        assert columns(c, table) == columns(fresh, table), table
    assert objects(c) == objects(fresh)


def test_v2_upgrade_makes_a_backup_once(tmp_path):
    db = make_v2_db(tmp_path / "old.db")
    init_db(db).close()
    backups = list((tmp_path / "backups").glob("old.v2.*.db"))
    assert len(backups) == 1
    b = sqlite3.connect(backups[0])
    assert b.execute("PRAGMA user_version").fetchone()[0] == 2
    assert b.execute("SELECT COUNT(*) FROM bib_detection").fetchone()[0] == 3
    b.close()
    init_db(db).close()                                   # already v3: nothing happens
    assert len(list((tmp_path / "backups").glob("*.db"))) == 1


def test_failed_upgrade_leaves_version_2_untouched(tmp_path, monkeypatch):
    db = make_v2_db(tmp_path / "old.db")

    def boom(sql):
        raise RuntimeError("simulated failure late in the upgrade")
    monkeypatch.setattr(migrations, "schema_statements", boom)
    with pytest.raises(RuntimeError):
        init_db(db)
    c = sqlite3.connect(db)
    assert c.execute("PRAGMA user_version").fetchone()[0] == 2
    assert "evidence_json" not in [r[1] for r in c.execute("PRAGMA table_info(bib_detection)")]
    assert "analysis_device" not in [r[1] for r in c.execute("PRAGMA table_info(photos)")]
    assert c.execute("SELECT COUNT(*) FROM bib_detection").fetchone()[0] == 3
    assert c.execute("SELECT name FROM sqlite_master WHERE name = 'v_photo_bibs'").fetchone()
    c.close()
    monkeypatch.undo()
    assert init_db(db).execute("PRAGMA user_version").fetchone()[0] == 3     # retry works


def test_newer_database_is_refused(tmp_path):
    db = tmp_path / "future.db"
    c = sqlite3.connect(db)
    c.execute("PRAGMA user_version = 4")
    c.close()
    with pytest.raises(SchemaError):
        init_db(db)


def test_center_crop_source_is_accepted(tmp_path):
    c = init_db(tmp_path / "t.db")
    c.execute("INSERT INTO events (id, name) VALUES (1, 'E')")
    c.execute("INSERT INTO photos (id, event_id, file_path, path_key, file_name) VALUES (1, 1, 'x', 'x', 'x')")
    c.execute("INSERT INTO bib_detection (photo_id, bib_number, source) VALUES (1, '123', 'center_crop')")
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("INSERT INTO bib_detection (photo_id, bib_number, source) VALUES (1, '1', 'banner')")


def test_schema_statement_splitter_handles_comments():
    sql = "-- a; comment\nCREATE TABLE a (x TEXT DEFAULT ';');\n-- trailing comment\n"
    assert migrations.schema_statements(sql) == ["-- a; comment\nCREATE TABLE a (x TEXT DEFAULT ';');"]
