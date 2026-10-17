"""Schema rules, checked against a real SQLite database file."""
import sqlite3

import pytest

from database.database import SCHEMA_VERSION, SchemaError, get_or_create_event, init_db


@pytest.fixture
def conn(tmp_path):
    c = init_db(tmp_path / "test.db")
    yield c
    c.close()


def add_photo(conn, name="a.jpg", event="Race"):
    event_id = get_or_create_event(conn, event)
    return conn.execute(
        "INSERT INTO photos (event_id, file_path, path_key, file_name) VALUES (?, ?, ?, ?)",
        (event_id, f"/p/{name}", f"/p/{name}".lower(), name)).lastrowid


def add_detection(conn, photo_id, bib, conf=0.8, source="person_body"):
    return conn.execute(
        "INSERT INTO bib_detection (photo_id, bib_number, confidence, source) VALUES (?, ?, ?, ?)",
        (photo_id, bib, conf, source)).lastrowid


def feedback(conn, photo_id, verdict, original=None, corrected=None):
    conn.execute("INSERT INTO detection_feedback (photo_id, verdict, original_bib, corrected_bib) "
                 "VALUES (?, ?, ?, ?)", (photo_id, verdict, original, corrected))


def reanalyse(conn, photo_id, *bibs):
    """What the pipeline does on re-analysis: replace all AI rows of a photo."""
    conn.execute("DELETE FROM bib_detection WHERE photo_id = ?", (photo_id,))
    for b in bibs:
        add_detection(conn, photo_id, b)


def bibs(conn, photo_id=None):
    sql = "SELECT photo_id, bib_number, verification FROM v_photo_bibs"
    rows = conn.execute(sql).fetchall()
    return sorted((r["bib_number"], r["verification"]) for r in rows
                  if photo_id is None or r["photo_id"] == photo_id)


# ---------------------------------------------------------------- structure

def test_tables_and_version(conn):
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert {"events", "event_bibs", "analysis_runs", "photos", "bib_detection",
            "detection_feedback", "v_photo_bibs"} <= names
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_photo_requires_existing_event(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO photos (event_id, file_path, path_key, file_name) "
                     "VALUES (999, 'x', 'x', 'x')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO photos (event_id, file_path, path_key, file_name) "
                     "VALUES (NULL, 'x', 'x', 'x')")


def test_empty_event_name_rejected(conn):
    with pytest.raises(ValueError):
        get_or_create_event(conn, "   ")


def test_same_path_once_per_event_but_same_name_in_other_folder_ok(conn):
    e = get_or_create_event(conn, "Race")
    ins = "INSERT INTO photos (event_id, file_path, path_key, file_name) VALUES (?, ?, ?, ?)"
    conn.execute(ins, (e, r"D:\A\IMG_1.JPG", r"d:\a\img_1.jpg", "IMG_1.JPG"))
    conn.execute(ins, (e, r"D:\B\IMG_1.JPG", r"d:\b\img_1.jpg", "IMG_1.JPG"))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(ins, (e, r"d:\a\IMG_1.jpg", r"d:\a\img_1.jpg", "IMG_1.jpg"))


def test_reopening_keeps_data(tmp_path):
    c = init_db(tmp_path / "x.db"); get_or_create_event(c, "Race"); c.close()
    c = init_db(tmp_path / "x.db")
    assert get_or_create_event(c, "Race") == 1


def test_upgrade_from_phase2_schema_keeps_events(tmp_path):
    db = tmp_path / "old.db"
    old = sqlite3.connect(db)
    old.executescript("""
        CREATE TABLE events (id INTEGER PRIMARY KEY, name TEXT UNIQUE, created_time TEXT);
        CREATE TABLE photos (id INTEGER PRIMARY KEY, event_id INTEGER);
        INSERT INTO events (name, created_time) VALUES ('Taihu 2026', '2026-10-13 10:00:00');
        PRAGMA user_version = 1;""")
    old.close()
    c = init_db(db)
    assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert [r["name"] for r in c.execute("SELECT name FROM events")] == ["Taihu 2026"]


def test_upgrade_refused_if_old_database_has_photos(tmp_path):
    db = tmp_path / "old.db"
    old = sqlite3.connect(db)
    old.executescript("""CREATE TABLE events (id INTEGER PRIMARY KEY, name TEXT, created_time TEXT);
        CREATE TABLE photos (id INTEGER PRIMARY KEY, event_id INTEGER);
        INSERT INTO photos (event_id) VALUES (1); PRAGMA user_version = 1;""")
    old.close()
    with pytest.raises(SchemaError):
        init_db(db)


# ---------------------------------------------------------------- bibs

def test_leading_zeros_are_stored_as_text(conn):
    p = add_photo(conn)
    add_detection(conn, p, "00123")
    add_detection(conn, p, "UT001")
    assert bibs(conn) == [("00123", "unverified"), ("UT001", "unverified")]
    row = conn.execute("SELECT typeof(bib_number) t FROM bib_detection LIMIT 1").fetchone()
    assert row["t"] == "text"


def test_multiple_bibs_per_photo_and_duplicates_merged(conn):
    p = add_photo(conn)
    add_detection(conn, p, "10258", 0.7)
    add_detection(conn, p, "10311", 0.9)
    add_detection(conn, p, "10258", 0.95, source="full_image")  # same runner, read twice
    rows = conn.execute("SELECT bib_number, confidence FROM v_photo_bibs ORDER BY bib_number").fetchall()
    assert [tuple(r) for r in rows] == [("10258", 0.95), ("10311", 0.9)]


def test_unreadable_regions_are_kept_but_not_searchable(conn):
    p = add_photo(conn)
    add_detection(conn, p, None)            # region found, OCR failed
    assert conn.execute("SELECT COUNT(*) FROM bib_detection").fetchone()[0] == 1
    assert bibs(conn) == []


# ---------------------------------------------------------------- feedback

def test_feedback_verdicts(conn):
    p = add_photo(conn)
    for b in ("10258", "10253", "2026", "777"):
        add_detection(conn, p, b)
    feedback(conn, p, "correct", "10258")
    feedback(conn, p, "incorrect", "2026")              # e.g. year on a banner
    feedback(conn, p, "corrected", "10253", "10255")
    feedback(conn, p, "added", None, "UT001")           # AI missed this runner
    assert bibs(conn) == [("10255", "corrected"), ("10258", "correct"),
                          ("777", "unverified"), ("UT001", "added")]


def test_feedback_survives_reanalysis(conn):
    p = add_photo(conn)
    add_detection(conn, p, "10253")
    add_detection(conn, p, "2026")
    feedback(conn, p, "corrected", "10253", "10255")
    feedback(conn, p, "incorrect", "2026")
    feedback(conn, p, "added", None, "UT001")
    # A new model run produces the same mistakes again, plus a new reading.
    reanalyse(conn, p, "10253", "2026", "10311")
    assert bibs(conn) == [("10255", "corrected"), ("10311", "unverified"), ("UT001", "added")]
    # ... and a run that finds nothing still keeps the photographer's bibs.
    reanalyse(conn, p)
    assert bibs(conn) == [("10255", "corrected"), ("UT001", "added")]


def test_feedback_applies_only_to_its_own_photo(conn):
    p1, p2 = add_photo(conn, "a.jpg"), add_photo(conn, "b.jpg")
    add_detection(conn, p1, "2026"); add_detection(conn, p2, "2026")
    feedback(conn, p1, "incorrect", "2026")
    assert bibs(conn, p1) == [] and bibs(conn, p2) == [("2026", "unverified")]


@pytest.mark.parametrize("verdict, original, corrected", [
    ("corrected", "10253", None), ("corrected", None, "10255"),
    ("added", "10253", "10255"), ("added", None, None),
    ("correct", None, None), ("incorrect", None, None), ("maybe", "1", "2"),
])
def test_invalid_feedback_rejected(conn, verdict, original, corrected):
    p = add_photo(conn)
    with pytest.raises(sqlite3.IntegrityError):
        feedback(conn, p, verdict, original, corrected)


def test_one_verdict_per_reading(conn):
    p = add_photo(conn)
    feedback(conn, p, "correct", "10258")
    with pytest.raises(sqlite3.IntegrityError):
        feedback(conn, p, "incorrect", "10258")
