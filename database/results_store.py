"""
Saving analysis results to the SQLite index (V0.2, schema 3).

Shared by live OCR runs ("index_cli.py ocr ... --event NAME") and CSV
imports ("index_cli.py import-csv ..."):

  * settings_hash()  - fingerprint of the settings that change OCR RESULTS
                       (mode, crop, thresholds, models, bib rules). Device,
                       thread count, debug images and the photo sample are
                       NOT part of it.
  * check_conflicts() - import protection. By default a run is refused when
                       (a) a photo already has results in this event, or
                       (b) the event already holds results made with other
                       settings (a different OCR experiment). --replace
                       allows both, explicitly.
  * OCRRunStore      - writes each photo in its own transaction as soon as
                       it is finished, so a stopped run keeps its photos.

Original photos are only read (metadata and fingerprint for new rows).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3

from database import database as db
from scanner.photo_scanner import UnreadableImageError, make_path_key, read_photo_info

# Keys of the diagnostic "settings" dict (ocr/diagnostic.py, report.json)
# that change which bibs are found. Missing keys count as None.
RESULT_SETTING_KEYS = (
    "processing_mode", "detector", "ocr_engine", "crop_strategy", "crop_fractions",
    "person_conf_threshold", "min_person_size_px", "full_image_fallback",
    "full_image_min_ocr_confidence", "full_image_require_person_overlap",
    "min_ocr_confidence", "enable_mkldnn", "text_det_limit_side_len", "text_det_limit_type",
    "fix_digit_confusions", "bib_length", "bib_max_letters", "bib_patterns",
    "registered_bibs", "center_crop", "ranking_weights",
)


class ResultConflict(Exception):
    """Refused: the run would overwrite or mix results in this event."""


def strip_device(text):
    """'yolo26n.pt (conf 0.25, imgsz 1280, device cpu)' -> without the device part."""
    if not isinstance(text, str):
        return text
    text = re.sub(r",\s*device [^,)]*", "", text)
    return re.sub(r"device [^,)]*,\s*", "", text)


def device_from_settings(settings: dict) -> str | None:
    m = re.search(r"device ([^,)]+)", str(settings.get("ocr_engine") or ""))
    return m.group(1).strip() if m else None


def result_settings(settings: dict) -> dict:
    out = {}
    for k in RESULT_SETTING_KEYS:
        v = settings.get(k)
        out[k] = strip_device(v) if k in ("detector", "ocr_engine") else v
    return json.loads(json.dumps(out, default=str))      # tuples -> lists, as in report.json


def settings_hash(settings: dict) -> str:
    blob = json.dumps(result_settings(settings), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ photos

def get_or_create_photo(conn: sqlite3.Connection, event_id: int, path: str) -> int:
    """
    The photos row for this file in this event, created if needed. New rows
    get metadata and fingerprint read from the file (read-only); files that
    no longer exist are recorded as 'missing', unreadable ones as 'unreadable'.
    """
    path = os.path.abspath(path)
    key = make_path_key(path)
    row = conn.execute("SELECT id FROM photos WHERE event_id = ? AND path_key = ?",
                       (event_id, key)).fetchone()
    if row:
        return row["id"]
    minimal = {"file_path": path, "path_key": key, "file_name": os.path.basename(path)}
    if not os.path.isfile(path):
        return db.insert_photo(conn, event_id, minimal, file_status="missing",
                               file_error="file not found when results were stored")
    try:
        meta = read_photo_info(path).as_db_dict()
    except UnreadableImageError as exc:
        return db.insert_photo(conn, event_id, minimal, file_status="unreadable",
                               file_error=str(exc))
    return db.insert_photo(conn, event_id, meta)


# ------------------------------------------------------------------ conflicts

def check_conflicts(conn: sqlite3.Connection, event_id: int, paths: list[str],
                    new_hash: str, replace: bool) -> list[str]:
    """
    Raise ResultConflict unless `replace`. Returns human-readable warnings
    (what --replace is about to overwrite or mix), empty if nothing.
    """
    keys = {make_path_key(os.path.abspath(p)) for p in paths}
    done = conn.execute(
        "SELECT path_key, file_path, analysis_settings_hash FROM photos "
        "WHERE event_id = ? AND analysis_status = 'done'", (event_id,)).fetchall()
    overlap = [r["file_path"] for r in done if r["path_key"] in keys]
    other_hashes = sorted({r["analysis_settings_hash"] for r in done
                           if r["analysis_settings_hash"] != new_hash}, key=str)
    problems = []
    if overlap:
        problems.append(f"{len(overlap)} photo(s) already have results in this event, e.g. "
                        + "; ".join(overlap[:3]))
    if other_hashes:
        problems.append("this event already contains results made with different OCR settings "
                        f"(settings hash {', '.join(map(str, other_hashes))}; this run: {new_hash})")
    if problems and not replace:
        raise ResultConflict(
            "Refused, nothing was changed:\n  - " + "\n  - ".join(problems)
            + "\nUse a separate event for a different OCR experiment, "
              "or add --replace to overwrite these results on purpose.")
    return problems


# ------------------------------------------------------------------ live OCR runs

class OCRRunStore:
    """Plugs into ocr.diagnostic.run_ocr_diagnostic(store=...)."""

    def __init__(self, conn: sqlite3.Connection, event_id: int, replace: bool = False):
        self.conn, self.event_id, self.replace = conn, event_id, replace
        self.run_id: int | None = None
        self.hash: str | None = None
        self.device: str | None = None
        self.saved = 0
        self.warnings: list[str] = []

    def begin(self, diag, selected: list[str]) -> None:
        self.hash = settings_hash(diag.settings)
        self.device = device_from_settings(diag.settings)
        self.warnings = check_conflicts(self.conn, self.event_id, selected, self.hash, self.replace)
        self.run_id = db.start_run(
            self.conn, self.event_id, source="ocr", source_folder=diag.folder,
            source_path=diag.run_dir, settings_json=json.dumps(diag.settings, default=str),
            settings_hash=self.hash, device=self.device,
            detector_model=diag.settings.get("detector"), ocr_model=diag.settings.get("ocr_engine"),
            photos_total=len(selected))

    def save(self, row) -> None:
        from ocr.photo_ocr import photo_bib_rows
        photo_id = get_or_create_photo(self.conn, self.event_id, row.file)
        if row.status == "ok" and row.result is not None:
            db.save_photo_result(self.conn, photo_id, self.run_id, photo_bib_rows(row.result),
                                 settings_hash=self.hash, device=self.device,
                                 processing_ms=row.total_ms)
        else:
            db.save_photo_result(self.conn, photo_id, self.run_id, [], error=row.error or row.status)
        self.saved += 1

    def end(self, completed: bool) -> None:
        if self.run_id is not None:
            db.finish_run(self.conn, self.run_id, "finished" if completed else "cancelled", self.saved)
