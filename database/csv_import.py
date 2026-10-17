"""
Import an existing OCR diagnostic batch (data/debug/ocr_<date>_<time>/)
into the SQLite index, under an explicitly named event.

Read from the batch folder (never written):
  photos.csv           optional; every photo of the run, including failures
                       and photos without a bib. Without it (e.g. a copied
                       photo_bibs.csv) only photos with a bib are imported.
  photo_bibs.csv       one row per (photo, bib)  -> bib_detection rows
  readings.csv         optional; only ACCEPTED readings are used, as the
                       small evidence_json of each bib
  report.json          optional; run settings (settings hash, device)
  review_overrides.csv optional; "reject" -> feedback 'incorrect',
                       "restore" -> nothing (the bib stays visible)

Rules:
  * The event name is required; the event is created if it does not exist.
  * The same batch (same photo_bibs.csv content) is imported once per
    event; importing it again does nothing.
  * Conflicts stop the import before anything is written (see
    results_store.check_conflicts); --replace overrides them.
  * The import is ONE transaction: it is stored completely or not at all.
  * Original photos are only read (metadata and fingerprint of new rows).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from database import database as db
from database.results_store import (ResultConflict, check_conflicts, device_from_settings,
                                    get_or_create_photo, settings_hash)
from ocr.numeric_candidates import numeric_identifier
from ocr.photo_ocr import MAX_EVIDENCE_READINGS, source_from_region_ids
from scanner.photo_scanner import make_path_key

__all__ = ["ImportSummary", "ResultConflict", "import_batch"]


@dataclass
class ImportSummary:
    batch_dir: str
    event_name: str
    event_id: int | None = None
    run_id: int | None = None
    already_imported: bool = False
    photos: int = 0
    photos_with_bibs: int = 0
    photos_failed: int = 0
    photo_files_missing: int = 0
    pairs: int = 0
    feedback_added: int = 0
    overrides_ignored: int = 0
    settings_hash: str = ""
    warnings: list[str] = field(default_factory=list)    # what --replace overwrote / mixed
    notes: list[str] = field(default_factory=list)       # incomplete batch folders


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _abs(file_value: str, batch: Path) -> str:
    p = Path((file_value or "").strip())
    return str(p if p.is_absolute() else (batch / p))


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v):
    f = _float(v)
    return int(round(f)) if f is not None else None


def _bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def import_batch(conn, batch_dir, event_name: str, replace: bool = False) -> ImportSummary:
    batch = Path(batch_dir).resolve()
    event_name = (event_name or "").strip()
    if not event_name:
        raise ValueError("An event name is required for a CSV import")
    photos_csv, bibs_csv = batch / "photos.csv", batch / "photo_bibs.csv"
    if not bibs_csv.is_file():
        raise FileNotFoundError(f"Not an OCR diagnostic batch, missing {bibs_csv}")

    summary = ImportSummary(str(batch), event_name)
    sha = hashlib.sha256(bibs_csv.read_bytes()).hexdigest()
    report = batch / "report.json"
    settings = json.loads(report.read_text(encoding="utf-8")).get("settings", {}) \
        if report.is_file() else {}
    summary.settings_hash = settings_hash(settings)
    device = device_from_settings(settings)

    bib_rows = _read_csv(bibs_csv)
    if photos_csv.is_file():
        photo_rows = _read_csv(photos_csv)
    else:   # e.g. a copied photo_bibs.csv only: the photos that have a bib
        photo_rows = [{"file": f, "status": "bib_read"}
                      for f in dict.fromkeys(r["file"] for r in bib_rows)]
        summary.notes.append("photos.csv not found: only photos with at least one bib "
                             "were imported (photos without a bib or failed are unknown)")
    if not report.is_file():
        summary.notes.append("report.json not found: OCR settings and device are unknown")
    paths = [_abs(r["file"], batch) for r in photo_rows]

    ev = conn.execute("SELECT id FROM events WHERE name = ?", (event_name,)).fetchone()
    if ev:
        summary.event_id = ev["id"]
        if conn.execute("SELECT 1 FROM imported_batches WHERE event_id = ? AND photo_bibs_sha256 = ?",
                        (ev["id"], sha)).fetchone():
            summary.already_imported = True
            return summary
        summary.warnings = check_conflicts(conn, ev["id"], paths, summary.settings_hash, replace)
    else:
        summary.event_id = db.get_or_create_event(conn, event_name)

    evidence = _accepted_readings(batch / "readings.csv", batch)
    bibs_by_photo: dict[str, list[dict]] = {}
    for r in bib_rows:
        bibs_by_photo.setdefault(make_path_key(_abs(r["file"], batch)), []).append(r)

    run_id = db.start_run(
        conn, summary.event_id, source="csv_import",
        source_folder=json.loads(report.read_text(encoding="utf-8")).get("folder", str(batch))
        if report.is_file() else str(batch),
        source_path=str(batch), settings_json=json.dumps(settings, ensure_ascii=False),
        settings_hash=summary.settings_hash, device=device,
        detector_model=settings.get("detector"), ocr_model=settings.get("ocr_engine"),
        photos_total=len(photo_rows))
    summary.run_id = run_id
    try:
        with conn:                                   # one transaction for the whole batch
            for prow, path in zip(photo_rows, paths):
                photo_id = get_or_create_photo(conn, summary.event_id, path)
                summary.photos += 1
                status = conn.execute("SELECT file_status FROM photos WHERE id = ?",
                                      (photo_id,)).fetchone()["file_status"]
                summary.photo_files_missing += status == "missing"
                if (prow.get("status") or "").startswith("failed"):
                    summary.photos_failed += 1
                    db.save_photo_result(conn, photo_id, run_id, [], commit=False,
                                         error=prow.get("error") or "failed in diagnostic run")
                    continue
                key = make_path_key(path)
                rows = [_detection_row(b, evidence.get((key, b["bib"]), []))
                        for b in bibs_by_photo.get(key, [])]
                summary.photos_with_bibs += bool(rows)
                summary.pairs += len(rows)
                db.save_photo_result(conn, photo_id, run_id, rows, commit=False,
                                     settings_hash=summary.settings_hash, device=device,
                                     processing_ms=_float(prow.get("total_ms")))
            summary.feedback_added, summary.overrides_ignored = _import_overrides(
                conn, batch, summary.event_id)
            conn.execute("INSERT INTO imported_batches (event_id, run_id, source_path, "
                         "photo_bibs_sha256, photos, pairs) VALUES (?, ?, ?, ?, ?, ?)",
                         (summary.event_id, run_id, str(batch), sha, summary.photos, summary.pairs))
    except BaseException:
        db.finish_run(conn, run_id, "failed", 0)
        raise
    db.finish_run(conn, run_id, "finished", summary.photos)
    return summary


def _detection_row(b: dict, readings: list[dict]) -> dict:
    bib = (b.get("bib") or "").strip()
    best = readings[0] if readings else None
    ev = {"readings": [
        {"raw_text": r["raw_text"], "ocr_confidence": _float(r["ocr_confidence"]),
         "region_id": r["region_id"], "region_source": r["region_source"],
         "box": [_int(r["line_x"]), _int(r["line_y"]), _int(r["line_w"]), _int(r["line_h"])],
         "letter_digit_fix": _bool(r.get("letter_digit_fix"))}
        for r in readings[:MAX_EVIDENCE_READINGS]] if readings else None,
        "imported_photo_no": b.get("photo_no")}
    conf = _float(b.get("best_confidence"))
    numeric = (b.get("numeric_identifier") or "").strip() or numeric_identifier(bib)
    ocr_text = best["raw_text"] if best else None
    source = (best["region_source"] if best and best.get("region_source")
              else source_from_region_ids(b.get("region_ids") or ""))
    from ocr.bib_filter import default_filter, filtered_evidence_json
    evidence_json, _ = filtered_evidence_json(default_filter(), json.dumps(ev, ensure_ascii=False),
                                              bib, numeric, conf, ocr_text or bib, source)
    return {
        "bib_number": bib,
        "numeric_bib": numeric,
        "confidence": conf, "ocr_confidence": conf, "det_confidence": None,
        "bbox_x": _int(b.get("best_box_x")), "bbox_y": _int(b.get("best_box_y")),
        "bbox_width": _int(b.get("best_box_w")), "bbox_height": _int(b.get("best_box_h")),
        "ocr_text": ocr_text,
        "source": source,
        "readings": _int(b.get("readings")), "region_ids": b.get("region_ids") or None,
        "letter_digit_fix": int(_bool(b.get("letter_digit_fix"))),
        "evidence_json": evidence_json,   # V0.3.1: includes the noise-filter result
    }


def _accepted_readings(path: Path, batch: Path) -> dict[tuple[str, str], list[dict]]:
    """(photo path_key, bib) -> accepted readings, best confidence first."""
    if not path.is_file():
        return {}
    out: dict[tuple[str, str], list[dict]] = {}
    for r in _read_csv(path):
        if _bool(r.get("accepted")) and r.get("candidate_bib"):
            out.setdefault((make_path_key(_abs(r["file"], batch)), r["candidate_bib"]), []).append(r)
    for rs in out.values():
        rs.sort(key=lambda r: -(_float(r.get("ocr_confidence")) or 0.0))
    return out


def _import_overrides(conn, batch: Path, event_id: int) -> tuple[int, int]:
    """review_overrides.csv of this batch -> feedback. Returns (added, ignored)."""
    path = batch / "review_overrides.csv"
    if not path.is_file():
        return 0, 0
    own = make_path_key(batch / "photo_bibs.csv")
    added = ignored = 0
    for r in _read_csv(path):
        action, bib = r.get("action"), (r.get("bib") or "").strip()
        if (not r.get("batch_id") or make_path_key(r["batch_id"]) != own
                or action not in ("reject", "restore") or not bib):
            ignored += 1
            continue
        if action == "restore":
            continue                   # the bib simply stays visible: nothing to store
        photo = conn.execute("SELECT id FROM photos WHERE event_id = ? AND path_key = ?",
                             (event_id, make_path_key(os.path.abspath(r.get("photo_identity") or ""))
                              )).fetchone()
        if photo is None:
            ignored += 1
            continue
        added += db.add_feedback(conn, photo["id"], "incorrect", original_bib=bib)
    return added, ignored
