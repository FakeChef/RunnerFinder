"""
Incremental folder scan: bring the database in line with a photo folder.

For each image file found:
  new path          -> read metadata; if its fingerprint matches a photo
                       whose old file no longer exists, it was MOVED or
                       renamed: update that row (keeps id, results and
                       manual corrections). Otherwise insert a new row.
  known, unchanged  -> (same size and modification time) nothing re-read
  known, changed    -> metadata re-read, marked for re-analysis
  unreadable        -> recorded with the error; the scan continues

After a COMPLETE walk, photos under the scanned folder that were not
found are marked 'missing'. Nothing is ever deleted, so corrections
survive, and a later scan that finds the file again restores it.

Progress is committed every `commit_every` files. If a scan is
interrupted, running it again skips everything already recorded.
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable

from config import ScanConfig
from database import database as db
from scanner.photo_scanner import (UnreadableImageError, is_under, iter_photo_files,
                                   make_path_key, read_photo_info)

ProgressCallback = Callable[[int, str], None]   # (files processed so far, current path)


@dataclass
class ScanSummary:
    found: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    relinked: int = 0          # moved/renamed files recognised by fingerprint
    unreadable: int = 0
    newly_missing: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)   # (path, message)
    elapsed_s: float = 0.0
    completed: bool = False


def sync_folder(conn: sqlite3.Connection, event_id: int, root: str | os.PathLike,
                cfg: ScanConfig | None = None, exclude_dirs: tuple = (),
                progress: ProgressCallback | None = None) -> ScanSummary:
    """
    Scan `root` and update the photos of `event_id`. Raises
    FileNotFoundError / NotADirectoryError if root is not an existing
    folder; in that case nothing is changed (an unplugged drive must not
    mark the whole event as missing).
    """
    cfg = cfg or ScanConfig()
    root = os.path.abspath(os.fspath(root))
    if not os.path.exists(root):
        raise FileNotFoundError(f"Folder not found: {root}")
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Not a folder: {root}")

    t0 = time.perf_counter()
    summary = ScanSummary()
    root_key = make_path_key(root)
    known = db.photos_index(conn, event_id)
    seen_ids: set[int] = set()
    unchanged_ids: list[int] = []

    def on_walk_error(path: str, exc: Exception) -> None:
        summary.errors.append((path, f"{type(exc).__name__}: {exc}"))

    try:
        for path, st in iter_photo_files(root, cfg.extensions, cfg.recursive,
                                         cfg.skip_dir_names, exclude_dirs, on_walk_error):
            summary.found += 1
            key = make_path_key(path)
            row = known.get(key)

            if (row is not None and row["file_status"] == "ok"
                    and row["file_size"] == st.st_size
                    and row["file_mtime_ns"] == st.st_mtime_ns):
                summary.unchanged += 1
                unchanged_ids.append(row["id"])
                seen_ids.add(row["id"])
            else:
                photo_id = _process_file(conn, event_id, path, st, row, cfg, summary)
                if photo_id is not None:
                    seen_ids.add(photo_id)

            if summary.found % cfg.commit_every == 0:
                db.mark_seen(conn, unchanged_ids)
                unchanged_ids.clear()
                conn.commit()
            if progress:
                progress(summary.found, path)

        # The walk finished: anything under root that was not seen is missing.
        missing = [r["id"] for k, r in known.items()
                   if r["id"] not in seen_ids and r["file_status"] != "missing"
                   and is_under(k, root_key) and not os.path.exists(r["file_path"])]
        db.mark_missing(conn, missing)
        summary.newly_missing = len(missing)
        summary.completed = True
    finally:
        db.mark_seen(conn, unchanged_ids)
        conn.commit()     # keep partial progress even if interrupted
        summary.elapsed_s = time.perf_counter() - t0
    return summary


def _process_file(conn, event_id, path, st, row, cfg, summary) -> int | None:
    """Handle a new, changed, previously missing or unreadable file."""
    try:
        meta = read_photo_info(path, st, cfg.fingerprint_chunk_bytes).as_db_dict()
    except UnreadableImageError as exc:
        return _record_unreadable(conn, event_id, path, st, row, str(exc), summary)
    except OSError as exc:   # e.g. permission denied, file vanished during the scan
        summary.errors.append((path, f"{type(exc).__name__}: {exc}"))
        return row["id"] if row is not None else None

    if row is not None:
        # Known path. Re-analyse only if the content actually changed.
        content_changed = row["fingerprint"] != meta["fingerprint"]
        db.update_photo(conn, row["id"], meta, reset_analysis=content_changed)
        if content_changed:
            summary.changed += 1
        else:
            summary.unchanged += 1     # e.g. reappeared, or only mtime differed
        return row["id"]

    # New path: was it moved here from a location that no longer exists?
    for candidate in db.find_by_fingerprint(conn, event_id, meta["fingerprint"]):
        if not os.path.exists(candidate["file_path"]):
            db.update_photo(conn, candidate["id"], meta, reset_analysis=False)
            summary.relinked += 1
            return candidate["id"]

    summary.new += 1
    return db.insert_photo(conn, event_id, meta)


def _record_unreadable(conn, event_id, path, st, row, message, summary) -> int:
    summary.unreadable += 1
    summary.errors.append((path, message))
    meta = {"file_path": path, "path_key": make_path_key(path),
            "file_name": os.path.basename(path), "file_size": st.st_size,
            # mtime is NOT stored, so the file is retried on the next scan
            "file_mtime_ns": None}
    if row is not None:
        db.update_photo(conn, row["id"], meta, file_status="unreadable",
                        file_error=message, reset_analysis=True)
        return row["id"]
    return db.insert_photo(conn, event_id, meta, file_status="unreadable", file_error=message)
