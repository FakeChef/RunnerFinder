"""
Schema upgrades for existing database files.

upgrade_v2_to_v3 (V0.2):
  * backs the file up first, with SQLite's online backup API, to
    <database folder>/backups/<name>.v2.<date>_<time>.db
  * then, in ONE transaction (all or nothing):
      - photos:             + analysis_settings_hash, analysis_device
      - analysis_runs:      + source, source_path, device, settings_hash
      - detection_feedback: + numeric_bib (filled for existing rows)
      - bib_detection:      rebuilt (source may now be 'center_crop';
                            + numeric_bib, readings, region_ids,
                            letter_digit_fix, evidence_json); rows are copied
      - imported_batches table, new indexes and the v_photo_bibs view are
        created from schema.sql
      - user_version = 3
Nothing is deleted. If any step fails, the database stays at version 2.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from ocr.numeric_candidates import numeric_identifier

SCHEMA_FILE = Path(__file__).with_name("schema.sql")

# Same definition as in schema.sql (a test checks that both agree).
_BIB_DETECTION_V3 = """
CREATE TABLE bib_detection_v3 (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id            INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    run_id              INTEGER REFERENCES analysis_runs(id) ON DELETE SET NULL,
    bib_number          TEXT    CHECK (bib_number IS NULL OR bib_number <> ''),
    numeric_bib         TEXT,
    confidence          REAL,
    det_confidence      REAL,
    ocr_confidence      REAL,
    bbox_x              INTEGER CHECK (bbox_x >= 0),
    bbox_y              INTEGER CHECK (bbox_y >= 0),
    bbox_width          INTEGER CHECK (bbox_width >= 0),
    bbox_height         INTEGER CHECK (bbox_height >= 0),
    person_bbox_x       INTEGER,
    person_bbox_y       INTEGER,
    person_bbox_width   INTEGER,
    person_bbox_height  INTEGER,
    ocr_text            TEXT,
    source              TEXT    NOT NULL
                        CHECK (source IN ('person_body', 'center_crop', 'full_image', 'bib_model')),
    readings            INTEGER,
    region_ids          TEXT,
    letter_digit_fix    INTEGER,
    evidence_json       TEXT,
    created_time        TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
)"""

_V2_DETECTION_COLUMNS = (
    "id, photo_id, run_id, bib_number, confidence, det_confidence, ocr_confidence, "
    "bbox_x, bbox_y, bbox_width, bbox_height, person_bbox_x, person_bbox_y, "
    "person_bbox_width, person_bbox_height, ocr_text, source, created_time")

_ADD_COLUMNS = (
    "ALTER TABLE photos ADD COLUMN analysis_settings_hash TEXT",
    "ALTER TABLE photos ADD COLUMN analysis_device TEXT",
    "ALTER TABLE analysis_runs ADD COLUMN source TEXT",
    "ALTER TABLE analysis_runs ADD COLUMN source_path TEXT",
    "ALTER TABLE analysis_runs ADD COLUMN device TEXT",
    "ALTER TABLE analysis_runs ADD COLUMN settings_hash TEXT",
    "ALTER TABLE detection_feedback ADD COLUMN numeric_bib TEXT",
)


def schema_statements(sql: str) -> list[str]:
    """Split a SQL script into complete statements (comments and strings safe)."""
    out, buf = [], ""
    for line in sql.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            out.append(buf.strip())
            buf = ""
    if buf.strip() and not all(l.strip().startswith("--") or not l.strip()
                               for l in buf.splitlines()):
        raise ValueError("schema.sql ends with an incomplete statement")
    return out


def backup_database(conn: sqlite3.Connection, db_path: Path | str, tag: str) -> Path:
    """Copy the open database to backups/<name>.<tag>.<timestamp>.db and return the path."""
    db_path = Path(db_path)
    folder = db_path.parent / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = folder / f"{db_path.stem}.{tag}.{stamp}.db"
    n = 1
    while target.exists():
        target = folder / f"{db_path.stem}.{tag}.{stamp}_{n}.db"
        n += 1
    dest = sqlite3.connect(target)
    try:
        conn.backup(dest)
    finally:
        dest.close()
    return target


def upgrade_v2_to_v3(conn: sqlite3.Connection, db_path: Path | str) -> Path:
    """Upgrade an open schema-2 database in place. Returns the backup file path."""
    if conn.execute("PRAGMA user_version").fetchone()[0] != 2:
        raise ValueError("upgrade_v2_to_v3 needs a schema 2 database")
    conn.commit()
    backup = backup_database(conn, db_path, "v2")

    old_isolation = conn.isolation_level
    conn.isolation_level = None                 # explicit BEGIN/COMMIT below
    conn.execute("PRAGMA foreign_keys = OFF")   # needed for the table rebuild; not
    try:                                        # changeable inside a transaction
        conn.execute("BEGIN IMMEDIATE")
        try:
            _upgrade_steps(conn)
            problems = conn.execute("PRAGMA foreign_key_check").fetchall()
            if problems:
                raise sqlite3.IntegrityError(f"foreign key check failed: {problems[:5]}")
            conn.execute("PRAGMA user_version = 3")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.isolation_level = old_isolation
    return backup


def _upgrade_steps(conn: sqlite3.Connection) -> None:
    # The view refers to bib_detection; it is recreated from schema.sql below.
    conn.execute("DROP VIEW IF EXISTS v_photo_bibs")
    for sql in _ADD_COLUMNS:
        conn.execute(sql)

    conn.execute(_BIB_DETECTION_V3)
    conn.execute(f"INSERT INTO bib_detection_v3 ({_V2_DETECTION_COLUMNS}) "
                 f"SELECT {_V2_DETECTION_COLUMNS} FROM bib_detection")
    conn.execute("DROP TABLE bib_detection")
    conn.execute("ALTER TABLE bib_detection_v3 RENAME TO bib_detection")

    for row_id, bib in conn.execute(
            "SELECT id, bib_number FROM bib_detection WHERE bib_number IS NOT NULL").fetchall():
        conn.execute("UPDATE bib_detection SET numeric_bib = ? WHERE id = ?",
                     (numeric_identifier(bib), row_id))
    for row_id, verdict, original, corrected in conn.execute(
            "SELECT id, verdict, original_bib, corrected_bib FROM detection_feedback").fetchall():
        bib = original if verdict in ("correct", "incorrect") else corrected
        conn.execute("UPDATE detection_feedback SET numeric_bib = ? WHERE id = ?",
                     (numeric_identifier(bib), row_id))

    # Indexes (old ones on bib_detection went with the old table), the new
    # imported_batches table and the view: all "IF NOT EXISTS" in schema.sql.
    for sql in schema_statements(SCHEMA_FILE.read_text(encoding="utf-8")):
        conn.execute(sql)
