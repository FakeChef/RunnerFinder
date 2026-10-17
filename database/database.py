"""
SQLite access layer.

Phase 3: connection handling, schema creation/upgrade, events, and the
photo-table operations used by the scanner.
Phase 6: detection, feedback and search helpers.

Only the Python standard library is used, so this module (and its tests)
works without the AI packages installed.
"""
from __future__ import annotations

import sqlite3
from contextlib import nullcontext
from pathlib import Path

SCHEMA_FILE = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 3


class SchemaError(RuntimeError):
    """The database file cannot be used with this version of the app."""


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open the database with sensible settings for a local desktop app."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)  # Streamlit uses threads
    conn.row_factory = sqlite3.Row             # access columns by name
    conn.execute("PRAGMA foreign_keys = ON")   # SQLite has them OFF by default
    conn.execute("PRAGMA journal_mode = WAL")  # UI can read while analysis writes
    return conn


def init_db(db_path: Path | str, video_extension: bool = True) -> sqlite3.Connection:
    """Create or upgrade the database and return an open connection.
    video_extension=False (V1.3 GUI): never add the video tables - opening the app does not write an
    existing schema-3 index; the analysis worker adds them (after a backup) when it first analyses."""
    conn = connect(db_path)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    backed_up = False
    if version == 0:
        _create_schema(conn)
    elif version == 1:
        _upgrade_from_v1(conn)
    elif version == 2:
        from database.migrations import upgrade_v2_to_v3
        upgrade_v2_to_v3(conn, db_path)     # backs the file up first
        backed_up = True
    elif version > SCHEMA_VERSION:
        conn.close()
        raise SchemaError(
            f"{db_path} was created by a newer version of Runner Photo Finder "
            f"(schema {version}); this version supports schema {SCHEMA_VERSION}.")
    # V1.3: additive video tables (user_version stays 3; backup first when the index has data)
    if video_extension:
        from database.video_store import ensure_video_extension
        ensure_video_extension(conn, db_path, backup=not backed_up)   # the v2 backup already holds everything
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def _upgrade_from_v1(conn: sqlite3.Connection) -> None:
    """
    Schema 1 was the Phase 2 skeleton, which could only create events.
    If no photos exist yet, rebuild with schema 2 and keep the events.
    Otherwise refuse rather than risk losing data.
    """
    photos = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    if photos:
        raise SchemaError("This database uses the Phase 2 test schema and already "
                          "contains photos. Rename it and start a new one.")
    events = conn.execute("SELECT name, created_time FROM events").fetchall()
    objects = conn.execute(
        "SELECT type, name FROM sqlite_master "
        "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'").fetchall()
    conn.execute("PRAGMA foreign_keys = OFF")
    for obj in objects:
        conn.execute(f'DROP {obj["type"].upper()} IF EXISTS "{obj["name"]}"')
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")
    _create_schema(conn)
    conn.executemany("INSERT INTO events (name, created_time) VALUES (?, ?)",
                     [(e["name"], e["created_time"]) for e in events])
    conn.commit()


# ------------------------------------------------------------------ events

def get_or_create_event(conn: sqlite3.Connection, name: str) -> int:
    """Return the id of the event with this name, creating it if needed."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Event name must not be empty")
    row = conn.execute("SELECT id FROM events WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO events (name) VALUES (?)", (name,))
    conn.commit()
    return cur.lastrowid


def list_events(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT e.id, e.name, e.created_time,
                  (SELECT COUNT(*) FROM photos p WHERE p.event_id = e.id) AS photo_count
           FROM events e ORDER BY e.created_time DESC, e.id DESC"""
    ).fetchall()


# ------------------------------------------------------------------ photos

_METADATA_COLUMNS = (
    "file_path", "path_key", "file_name", "file_size", "file_mtime_ns", "fingerprint",
    "width", "height", "exif_orientation", "capture_time", "capture_offset",
    "capture_time_source", "camera_make", "camera_model", "camera_serial",
)


def photos_index(conn: sqlite3.Connection, event_id: int) -> dict[str, sqlite3.Row]:
    """All photos of an event, keyed by path_key (used by the scanner)."""
    rows = conn.execute(
        "SELECT id, path_key, file_path, file_size, file_mtime_ns, file_status, fingerprint "
        "FROM photos WHERE event_id = ?", (event_id,)).fetchall()
    return {r["path_key"]: r for r in rows}


def find_by_fingerprint(conn: sqlite3.Connection, event_id: int,
                        fingerprint: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, file_path, path_key, file_status FROM photos "
        "WHERE event_id = ? AND fingerprint = ? ORDER BY id",
        (event_id, fingerprint)).fetchall()


def insert_photo(conn: sqlite3.Connection, event_id: int, meta: dict,
                 file_status: str = "ok", file_error: str | None = None) -> int:
    cols = ["event_id", *_METADATA_COLUMNS, "file_status", "file_error", "last_seen_time"]
    values = [event_id, *(meta.get(c) for c in _METADATA_COLUMNS), file_status, file_error]
    sql = (f"INSERT INTO photos ({', '.join(cols)}) VALUES "
           f"({', '.join('?' * (len(cols) - 1))}, datetime('now', 'localtime'))")
    return conn.execute(sql, values).lastrowid


def update_photo(conn: sqlite3.Connection, photo_id: int, meta: dict, *,
                 file_status: str = "ok", file_error: str | None = None,
                 reset_analysis: bool) -> None:
    """
    Refresh a photo's file information (after a change or a move).
    reset_analysis=True marks it for re-analysis. Existing detections and
    manual feedback are NOT deleted here.
    """
    sets = [f"{c} = ?" for c in _METADATA_COLUMNS if c in meta]
    values = [meta[c] for c in _METADATA_COLUMNS if c in meta]
    sets += ["file_status = ?", "file_error = ?", "last_seen_time = datetime('now', 'localtime')"]
    values += [file_status, file_error]
    if reset_analysis:
        sets += ["analysis_status = 'pending'", "analysis_error = NULL"]
    conn.execute(f"UPDATE photos SET {', '.join(sets)} WHERE id = ?", [*values, photo_id])


def mark_seen(conn: sqlite3.Connection, photo_ids: list[int]) -> None:
    conn.executemany(
        "UPDATE photos SET file_status = 'ok', file_error = NULL, "
        "last_seen_time = datetime('now', 'localtime') WHERE id = ?",
        [(i,) for i in photo_ids])


def mark_missing(conn: sqlite3.Connection, photo_ids: list[int]) -> None:
    """Flag photos whose files were not found. Rows and feedback are kept."""
    conn.executemany("UPDATE photos SET file_status = 'missing' WHERE id = ?",
                     [(i,) for i in photo_ids])


# ------------------------------------------------------------ analysis results
#
# Since schema 3 the pipeline stores one bib_detection row per (photo, bib):
# the deduplicated result of photo_bibs.csv, plus a small evidence_json.
# Rejected OCR readings stay in the diagnostic CSV files only.

DETECTION_COLUMNS = (
    "bib_number", "numeric_bib", "confidence", "det_confidence", "ocr_confidence",
    "bbox_x", "bbox_y", "bbox_width", "bbox_height", "ocr_text", "source",
    "readings", "region_ids", "letter_digit_fix", "evidence_json",
)


def start_run(conn: sqlite3.Connection, event_id: int, *, source: str, source_folder: str,
              source_path: str | None = None, settings_json: str | None = None,
              settings_hash: str | None = None, device: str | None = None,
              detector_model: str | None = None, ocr_model: str | None = None,
              photos_total: int | None = None) -> int:
    """Create an analysis_runs row with status 'running' and return its id."""
    cur = conn.execute(
        "INSERT INTO analysis_runs (event_id, source_folder, status, detector_model, ocr_model, "
        "settings_json, photos_total, source, source_path, device, settings_hash) "
        "VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?)",
        (event_id, source_folder, detector_model, ocr_model, settings_json, photos_total,
         source, source_path, device, settings_hash))
    conn.commit()
    return cur.lastrowid


def finish_run(conn: sqlite3.Connection, run_id: int, status: str, photos_processed: int) -> None:
    """status: 'finished', 'cancelled' or 'failed'."""
    conn.execute("UPDATE analysis_runs SET status = ?, photos_processed = ?, "
                 "finished_time = datetime('now', 'localtime') WHERE id = ?",
                 (status, photos_processed, run_id))
    conn.commit()


def save_photo_result(conn: sqlite3.Connection, photo_id: int, run_id: int, rows: list[dict], *,
                      error: str | None = None, settings_hash: str | None = None,
                      device: str | None = None, processing_ms: float | None = None,
                      commit: bool = True) -> None:
    """
    Store one photo's analysis in ONE transaction (all or nothing).

    Success (error is None): the photo's previous bib_detection rows are
    replaced by `rows` (dicts with DETECTION_COLUMNS keys) and the photo is
    marked 'done'. Failure: the photo is marked 'error' and its previous
    results are kept. Manual feedback is never touched.
    commit=False: the caller owns a larger transaction (CSV import).
    """
    ms = round(processing_ms) if processing_ms is not None else None
    with conn if commit else nullcontext():   # commit on success, roll back on any exception
        if error is not None:
            conn.execute("UPDATE photos SET analysis_status = 'error', analysis_error = ? "
                         "WHERE id = ?", (error, photo_id))
            return
        conn.execute("DELETE FROM bib_detection WHERE photo_id = ?", (photo_id,))
        conn.executemany(
            f"INSERT INTO bib_detection (photo_id, run_id, {', '.join(DETECTION_COLUMNS)}) "
            f"VALUES (?, ?, {', '.join('?' * len(DETECTION_COLUMNS))})",
            [(photo_id, run_id, *(r.get(c) for c in DETECTION_COLUMNS)) for r in rows])
        conn.execute(
            "UPDATE photos SET analysis_status = 'done', analysis_error = NULL, processing_ms = ?, "
            "run_id = ?, analysis_settings_hash = ?, analysis_device = ? WHERE id = ?",
            (ms, run_id, settings_hash, device, photo_id))


def add_feedback(conn: sqlite3.Connection, photo_id: int, verdict: str,
                 original_bib: str | None = None, corrected_bib: str | None = None) -> bool:
    """
    Record a manual verdict unless one already exists for this (photo, bib).
    Returns True if a row was added. An existing verdict is never overwritten.
    """
    from ocr.numeric_candidates import numeric_identifier
    bib = original_bib if verdict in ("correct", "incorrect") else corrected_bib
    cur = conn.execute(
        "INSERT OR IGNORE INTO detection_feedback "
        "(photo_id, verdict, original_bib, corrected_bib, numeric_bib) VALUES (?, ?, ?, ?, ?)",
        (photo_id, verdict, original_bib, corrected_bib, numeric_identifier(bib)))
    return cur.rowcount == 1


def reject_bib(conn: sqlite3.Connection, photo_id: int, bib: str) -> None:
    """Search 'reject': this AI bib is wrong for this photo (feedback 'incorrect')."""
    from ocr.numeric_candidates import numeric_identifier
    with conn:
        conn.execute("DELETE FROM detection_feedback WHERE photo_id = ? AND original_bib = ?",
                     (photo_id, bib))
        conn.execute("INSERT INTO detection_feedback (photo_id, verdict, original_bib, numeric_bib) "
                     "VALUES (?, 'incorrect', ?, ?)", (photo_id, bib, numeric_identifier(bib)))


def restore_bib(conn: sqlite3.Connection, photo_id: int, bib: str) -> None:
    """Search 'restore': remove the 'incorrect' verdict; the AI bib is visible again."""
    with conn:
        conn.execute("DELETE FROM detection_feedback WHERE photo_id = ? AND original_bib = ? "
                     "AND verdict = 'incorrect'", (photo_id, bib))
