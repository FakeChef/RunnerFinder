"""
Incremental, resumable analysis of one photo folder into one event (V0.2 Phase 4).

    index_cli.py analyze "D:\\RacePhotos\\Race" --event "2026 Race"

Every run:
  1. scans the folder (scanner/folder_sync.py, already incremental): new
     photos are added, photos whose content changed are marked 'pending',
     moved/renamed photos keep their results, deleted ones become 'missing';
  2. selects, under that folder only, the photos that need OCR:
        new      analysis_status 'pending', never analysed
        changed  analysis_status 'pending', analysed before (content changed)
        failed   analysis_status 'error' (retried on every run)
     Photos already 'done' are skipped; unreadable and missing files too.
     With --replace, photos done with OTHER OCR settings are added as well;
  3. runs the unchanged OCR pipeline on exactly those photos, saving every
     photo to SQLite as soon as it is finished (one transaction per photo).

Interruption (Ctrl+C, crash, power loss): finished photos are already
committed; the photo in progress was not, so it is still 'pending'. Running
the same command again continues with the remaining photos. Re-analysis
replaces a photo's rows, so there are never duplicate results. A run left as
'running' by a crash is marked 'cancelled' at the next start.

Not done on purpose (personal tool): no locking between two simultaneous
runs on the same database, no retry limits or scheduling.
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable

from config import AppConfig
from database import database as db
from scanner.folder_sync import ScanSummary, sync_folder
from scanner.photo_scanner import is_under, make_path_key


@dataclass
class Selection:
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    other_settings: list[str] = field(default_factory=list)   # only used with --replace
    done: int = 0
    unreadable: int = 0
    missing: int = 0

    @property
    def todo(self) -> list[str]:
        return [*self.new, *self.changed, *self.failed, *self.other_settings]


@dataclass
class AnalyzeSummary:
    folder: str
    event_name: str
    event_id: int | None = None
    scan: ScanSummary | None = None
    selection: Selection | None = None
    stale_runs_cancelled: int = 0
    run_id: int | None = None
    run_dir: str = ""
    settings_hash: str = ""
    device: str | None = None
    to_process: int = 0
    processed: int = 0
    failed_now: int = 0
    interrupted: bool = False
    thumbnails: int = 0          # Search-page thumbnails available for analysed photos (V0.3.1)
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    pipeline_used: str = ""      # V1.3: s3 | prefetch | sequential, as actually run
    pipeline_fallback: str = ""  # V1.3: reason when S3 was requested but not used


def select_photos(conn: sqlite3.Connection, event_id: int, root: str,
                  current_hash: str | None = None) -> Selection:
    """Photos of the event under `root` and what they need (see module docstring)."""
    root_key = make_path_key(root)
    sel = Selection()
    rows = conn.execute(
        "SELECT file_path, path_key, file_status, analysis_status, run_id, analysis_settings_hash "
        "FROM photos WHERE event_id = ? ORDER BY capture_time, file_name, id", (event_id,)).fetchall()
    for r in rows:
        if not is_under(r["path_key"], root_key):
            continue
        if r["file_status"] == "unreadable":
            sel.unreadable += 1
        elif r["file_status"] == "missing":
            sel.missing += 1
        elif r["analysis_status"] == "pending":
            (sel.new if r["run_id"] is None else sel.changed).append(r["file_path"])
        elif r["analysis_status"] == "error":
            sel.failed.append(r["file_path"])
        elif current_hash is not None and r["analysis_settings_hash"] != current_hash:
            sel.other_settings.append(r["file_path"])
        else:
            sel.done += 1
    return sel


def _thumbnail_hook(conn: sqlite3.Connection, s: "AnalyzeSummary", cfg: AppConfig):
    """on_decoded callback: save the Search-page thumbnail from the image already
    decoded and rotated for OCR (V0.3.1). Never decodes the photo again."""
    from utils.thumbnails import save_from_image
    fingerprints = {r["path_key"]: r["fingerprint"] for r in conn.execute(
        "SELECT path_key, fingerprint FROM photos WHERE event_id = ? AND fingerprint IS NOT NULL",
        (s.event_id,))}

    def hook(path: str, image) -> None:
        fp = fingerprints.get(make_path_key(path))
        if fp and save_from_image(image, cfg.paths.thumbnail_cache, fp, cfg.thumbnails):
            s.thumbnails += 1
    return hook


def cancel_stale_runs(conn: sqlite3.Connection, event_id: int) -> int:
    """Runs still 'running' were left by a crash or power loss (no locking: one run at a time)."""
    cur = conn.execute("UPDATE analysis_runs SET status = 'cancelled', "
                       "finished_time = datetime('now', 'localtime') "
                       "WHERE event_id = ? AND status = 'running'", (event_id,))
    conn.commit()
    return cur.rowcount


def analyze_folder(conn: sqlite3.Connection, folder, event_name: str, cfg: AppConfig,
                   load_models: Callable[[], tuple], *, replace: bool = False,
                   limit: int | None = None, debug: bool = False,
                   progress: Callable[[int, int, str], None] | None = None,
                   scan_progress: Callable[[int, str], None] | None = None,
                   prefetch: int = 0, three_stage: bool = False) -> AnalyzeSummary:
    """
    load_models() -> (person detector or None, OCR engine); called only if there
    is something to analyse. Raises FileNotFoundError for a missing folder and
    results_store.ResultConflict when the event holds results made with other
    OCR settings (unless replace). Ctrl+C is caught: summary.interrupted.
    prefetch (V1.2): photos decoded ahead in a producer thread (0 = off); the
    results, the commit order and cancel/resume behaviour do not change.
    three_stage (V1.3, with prefetch > 0): person detection also runs ahead in its own thread;
    results, commit order and cancel/resume behaviour do not change either.
    """
    from database.results_store import OCRRunStore, settings_hash
    from ocr.diagnostic import run_ocr_diagnostic, run_settings

    t0 = time.perf_counter()
    root = os.path.abspath(os.fspath(folder))
    for skipped in (cfg.paths.data_dir, cfg.paths.default_export_root):
        if is_under(make_path_key(root), make_path_key(skipped)):
            raise ValueError(f"The photo folder {root} is inside {skipped}, which the scanner always "
                             "skips (app data / export). Use a folder outside it.")
    s = AnalyzeSummary(root, event_name)
    s.event_id = db.get_or_create_event(conn, event_name)
    s.stale_runs_cancelled = cancel_stale_runs(conn, s.event_id)
    s.scan = sync_folder(conn, s.event_id, root, cfg.scan,
                         exclude_dirs=(cfg.paths.default_export_root, cfg.paths.data_dir),
                         progress=scan_progress)
    s.selection = select_photos(conn, s.event_id, root)
    if not s.selection.todo and not replace:
        s.elapsed_s = time.perf_counter() - t0
        return s                                     # nothing to do: models are not loaded

    detector, engine = load_models()
    s.settings_hash = settings_hash(run_settings(cfg, detector, engine, debug=debug))
    if replace:
        s.selection = select_photos(conn, s.event_id, root, s.settings_hash)
    todo = s.selection.todo[:limit] if limit else s.selection.todo
    s.to_process = len(todo)
    if not todo:
        s.elapsed_s = time.perf_counter() - t0
        return s

    store = OCRRunStore(conn, s.event_id, replace=replace)
    on_decoded = _thumbnail_hook(conn, s, cfg) if cfg.thumbnails.generate_during_analyze else None
    info: dict = {}
    try:
        diag = run_ocr_diagnostic(root, cfg, detector, engine, limit=None, debug=debug,
                                  progress=progress, store=store, files=todo,
                                  on_decoded=on_decoded, prefetch=prefetch, three_stage=three_stage,
                                  pipeline_info=info)
        s.run_dir = diag.run_dir
    except KeyboardInterrupt:
        s.interrupted = True
    s.pipeline_used, s.pipeline_fallback = info.get("pipeline_used", ""), info.get("pipeline_fallback", "")
    s.run_id, s.device, s.processed, s.warnings = store.run_id, store.device, store.saved, store.warnings
    todo_keys = {make_path_key(p) for p in todo}
    s.failed_now = sum(1 for r in conn.execute(
        "SELECT path_key FROM photos WHERE event_id = ? AND analysis_status = 'error'", (s.event_id,))
        if r["path_key"] in todo_keys)
    s.elapsed_s = time.perf_counter() - t0
    return s
