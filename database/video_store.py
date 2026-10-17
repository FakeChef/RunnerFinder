"""
V1.3: video extension of the index (database/video_schema.sql) and its read/write helpers.

ensure_video_extension(conn, db_path)
    Adds the video tables to a schema-3 index if they are missing:
      1. backup (SQLite online backup) to <db folder>/backups/<name>.pre_video.<time>.db,
         only if the index already holds events (a brand-new file needs none);
      2. ONE transaction: the tables of video_schema.sql + schema_extensions('video', 1);
         foreign-key and integrity checks before COMMIT; any failure rolls back.
    Existing tables are never touched and PRAGMA user_version stays 3 (RunnerFinder 1.2 refuses a
    higher version; it simply ignores the new tables), so a rollback to 1.2 keeps working.
    Idempotent: does nothing when the extension is present.

Results are written per video in one transaction (save_video_result), like photos.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

VIDEO_SCHEMA_FILE = Path(__file__).with_name("video_schema.sql")
EXTENSION = "video"
EXTENSION_VERSION = 1
VIDEO_TABLES = ("schema_extensions", "videos", "video_candidates", "video_feedback")


def extension_version(conn: sqlite3.Connection) -> int:
    """0 if the video extension is not installed."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_extensions'"
                        ).fetchone():
        return 0
    row = conn.execute("SELECT version FROM schema_extensions WHERE name = ?", (EXTENSION,)).fetchone()
    return int(row[0]) if row else 0


def ensure_video_extension(conn: sqlite3.Connection, db_path: Path | str | None,
                           backup: bool = True) -> Path | None:
    """Install the video extension if missing. Returns the backup path (None: no backup needed/made).
    backup=False: the caller has just made a full backup of this file (schema 2 -> 3 upgrade)."""
    version = extension_version(conn)
    if version >= EXTENSION_VERSION:
        return None
    from database.migrations import backup_database, schema_statements   # not at import: the GUI imports this
                                                                          # module and must not load ocr.*
    conn.commit()
    has_data = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] > 0
    made = None
    if backup and has_data and db_path is not None and str(db_path) != ":memory:":
        made = backup_database(conn, db_path, "pre_video")
    old_isolation = conn.isolation_level
    conn.isolation_level = None                      # explicit BEGIN / COMMIT
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for sql in schema_statements(VIDEO_SCHEMA_FILE.read_text(encoding="utf-8")):
                conn.execute(sql)
            conn.execute("INSERT OR REPLACE INTO schema_extensions (name, version) VALUES (?, ?)",
                         (EXTENSION, EXTENSION_VERSION))
            problems = conn.execute("PRAGMA foreign_key_check").fetchall()
            if problems:
                raise sqlite3.IntegrityError(f"foreign key check failed: {problems[:5]}")
            ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if ok != "ok":
                raise sqlite3.IntegrityError(f"integrity check failed: {ok}")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.isolation_level = old_isolation
    return made


# ------------------------------------------------------------------ videos of an event

def videos_index(conn: sqlite3.Connection, event_id: int) -> dict[str, sqlite3.Row]:
    rows = conn.execute("SELECT id, path_key, file_path, file_size, file_mtime_ns, file_status, fingerprint, "
                        "analysis_status FROM videos WHERE event_id = ?", (event_id,)).fetchall()
    return {r["path_key"]: r for r in rows}


_META = ("file_path", "path_key", "file_name", "file_size", "file_mtime_ns", "fingerprint",
         "duration_s", "fps", "frame_count", "width", "height")


def insert_video(conn: sqlite3.Connection, event_id: int, meta: dict, file_status: str = "ok",
                 file_error: str | None = None) -> int:
    cols = ["event_id", *_META, "file_status", "file_error", "last_seen_time"]
    vals = [event_id, *(meta.get(c) for c in _META), file_status, file_error]
    return conn.execute(f"INSERT INTO videos ({', '.join(cols)}) VALUES ({', '.join('?' * (len(cols) - 1))}, "
                        "datetime('now', 'localtime'))", vals).lastrowid


def update_video(conn: sqlite3.Connection, video_id: int, meta: dict, *, file_status: str = "ok",
                 file_error: str | None = None, reset_analysis: bool) -> None:
    sets = [f"{c} = ?" for c in _META if c in meta]
    vals = [meta[c] for c in _META if c in meta]
    sets += ["file_status = ?", "file_error = ?", "last_seen_time = datetime('now', 'localtime')"]
    vals += [file_status, file_error]
    if reset_analysis:
        sets += ["analysis_status = 'pending'", "analysis_error = NULL"]
    conn.execute(f"UPDATE videos SET {', '.join(sets)} WHERE id = ?", [*vals, video_id])


def save_video_result(conn: sqlite3.Connection, video_id: int, run_id: int | None, result: dict | None, *,
                      error: str | None = None, settings_hash: str | None = None,
                      processing_ms: float | None = None) -> None:
    """One video in ONE transaction. error: status 'error', previous results kept.
    result: dict from video.recognize.RecognitionResult.as_db()."""
    ms = round(processing_ms) if processing_ms is not None else None
    with conn:
        if error is not None:
            conn.execute("UPDATE videos SET analysis_status = 'error', analysis_error = ?, processing_ms = ?, "
                         "run_id = ? WHERE id = ?", (error, ms, run_id, video_id))
            return
        conn.execute("DELETE FROM video_candidates WHERE video_id = ?", (video_id,))
        conn.executemany(
            "INSERT INTO video_candidates (video_id, bib_number, numeric_bib, role, votes, frames, "
            "best_confidence, forms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(video_id, c["bib_number"], c["numeric_bib"], c["role"], c["votes"], c["frames"],
              c.get("best_confidence"), c.get("forms")) for c in result["candidates"]])
        conn.execute(
            "UPDATE videos SET analysis_status = 'done', analysis_error = NULL, recognition_status = ?, "
            "strategy = ?, frames_used = ?, escalated = ?, sample_times_json = ?, evidence_json = ?, "
            "thumbnail_fingerprint = ?, processing_ms = ?, run_id = ?, analysis_settings_hash = ? WHERE id = ?",
            (result["status"], result["strategy"], result["frames_used"], int(result["escalated"]),
             json.dumps(result["sample_times"]), json.dumps(result["evidence"], ensure_ascii=False),
             result.get("thumbnail_fingerprint"), ms, run_id, settings_hash, video_id))


# ------------------------------------------------------------------ review (search page)

def reject_video_bib(conn: sqlite3.Connection, video_id: int, numeric_bib: str) -> None:
    with conn:
        conn.execute("INSERT OR IGNORE INTO video_feedback (video_id, numeric_bib, verdict) "
                     "VALUES (?, ?, 'incorrect')", (video_id, numeric_bib))


def restore_video_bib(conn: sqlite3.Connection, video_id: int, numeric_bib: str) -> None:
    with conn:
        conn.execute("DELETE FROM video_feedback WHERE video_id = ? AND numeric_bib = ? AND verdict = 'incorrect'",
                     (video_id, numeric_bib))
