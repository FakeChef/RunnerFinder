"""
V1.3: incremental, resumable video analysis of one folder into one event (photos: indexing/analyze.py).

    sync_videos()     find .mp4/.mov/.m4v under the folder (same skip rules as photos), record new and
                      changed files (metadata probe), relink moved files by fingerprint, mark missing ones
    analyze_videos()  pending/changed/failed videos: sample frames, unchanged PhotoOCR + noise filter,
                      Adaptive S5->S7 decision (video/recognize.py), representative-frame thumbnail,
                      ONE transaction per video (database/video_store.save_video_result)

Originals are only opened for reading (OpenCV); nothing is written next to them.
Cancel: the `progress` callback may raise KeyboardInterrupt before each video; the video in progress
is not committed, so it stays 'pending' and the next run continues with it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable

from database import database as db
from database import video_store as vs
from scanner.photo_scanner import compute_fingerprint, is_under, iter_photo_files, make_path_key
from video import recognize as rz
from video.frames import VIDEO_EXTENSIONS, UnreadableVideoError, VideoReader, frame_index, probe


@dataclass
class VideoScanSummary:
    found: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    relinked: int = 0
    unreadable: int = 0
    newly_missing: int = 0


@dataclass
class VideoAnalyzeSummary:
    scan: VideoScanSummary = field(default_factory=VideoScanSummary)
    to_process: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    verified: int = 0
    ambiguous: int = 0
    unindexed: int = 0
    escalated: int = 0
    frames: int = 0
    run_id: int | None = None
    interrupted: bool = False
    elapsed_s: float = 0.0
    seconds_per_video: float | None = None


def iter_video_files(root: str, cfg) -> list[tuple[str, os.stat_result]]:
    return list(iter_photo_files(root, VIDEO_EXTENSIONS, cfg.scan.recursive, cfg.scan.skip_dir_names,
                                 (cfg.paths.default_export_root, cfg.paths.data_dir)))


def count_videos(root: str, cfg) -> int:
    return sum(1 for _ in iter_photo_files(root, VIDEO_EXTENSIONS, cfg.scan.recursive, cfg.scan.skip_dir_names,
                                           (cfg.paths.default_export_root, cfg.paths.data_dir)))


def _meta(path: str, st: os.stat_result) -> dict:
    info = probe(path)
    return {"file_path": path, "path_key": make_path_key(path), "file_name": os.path.basename(path),
            "file_size": st.st_size, "file_mtime_ns": st.st_mtime_ns,
            "fingerprint": compute_fingerprint(path, st.st_size), "duration_s": info.duration_s,
            "fps": info.fps, "frame_count": info.frame_count, "width": info.width, "height": info.height}


def sync_videos(conn: sqlite3.Connection, event_id: int, root: str, cfg) -> VideoScanSummary:
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Folder not found: {root}")
    s = VideoScanSummary()
    known = vs.videos_index(conn, event_id)
    seen: set[int] = set()
    for path, st in iter_video_files(root, cfg):
        s.found += 1
        key = make_path_key(path)
        row = known.get(key)
        if row is not None and row["file_size"] == st.st_size and row["file_mtime_ns"] == st.st_mtime_ns \
                and row["file_status"] != "unreadable":
            conn.execute("UPDATE videos SET file_status = 'ok', file_error = NULL, "
                         "last_seen_time = datetime('now', 'localtime') WHERE id = ?", (row["id"],))
            s.unchanged += 1
            seen.add(row["id"])
            continue
        try:
            meta = _meta(path, st)
            status, error = "ok", None
        except (UnreadableVideoError, OSError) as exc:
            meta = {"file_path": path, "path_key": key, "file_name": os.path.basename(path),
                    "file_size": st.st_size, "file_mtime_ns": st.st_mtime_ns}
            status, error = "unreadable", str(exc)
            s.unreadable += 1
        if row is not None:
            vs.update_video(conn, row["id"], meta, file_status=status, file_error=error, reset_analysis=True)
            s.changed += 1
            seen.add(row["id"])
            continue
        moved = None
        if meta.get("fingerprint"):
            for r in conn.execute("SELECT id, file_path FROM videos WHERE event_id = ? AND fingerprint = ? "
                                  "ORDER BY id", (event_id, meta["fingerprint"])).fetchall():
                if r["id"] not in seen and not os.path.exists(r["file_path"]):
                    moved = r["id"]
                    break
        if moved is not None:                       # moved / renamed: keep results and reviews
            vs.update_video(conn, moved, meta, file_status=status, file_error=error, reset_analysis=False)
            s.relinked += 1
            seen.add(moved)
        else:
            seen.add(vs.insert_video(conn, event_id, meta, status, error))
            s.new += 1
    root_key = make_path_key(root)
    for row in known.values():
        if row["id"] not in seen and is_under(row["path_key"], root_key) and row["file_status"] != "missing":
            conn.execute("UPDATE videos SET file_status = 'missing' WHERE id = ?", (row["id"],))
            s.newly_missing += 1
    conn.commit()
    return s


def select_videos(conn: sqlite3.Connection, event_id: int, root: str, settings_hash: str) -> tuple[list, int]:
    """(rows to analyse, done count). Pending, failed, and done with other settings."""
    root_key = make_path_key(root)
    todo, done = [], 0
    for r in conn.execute("SELECT * FROM videos WHERE event_id = ? ORDER BY file_name, id", (event_id,)):
        if not is_under(r["path_key"], root_key) or r["file_status"] != "ok":
            continue
        if r["analysis_status"] == "done" and r["analysis_settings_hash"] == settings_hash:
            done += 1
        else:
            todo.append(r)
    return todo, done


def video_settings_hash(photo_settings_hash: str, years: tuple[str, ...]) -> str:
    import hashlib
    blob = json.dumps({"strategy": rz.STRATEGY, "photo": photo_settings_hash, "s5": rz.S5_POSITIONS,
                       "esc": rz.ESCALATION_POSITIONS, "thr": [rz.S5_THRESHOLD, rz.S7_THRESHOLD],
                       "years": list(years)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class VideoRecognizer:
    """Frames -> unchanged PhotoOCR -> noise filter -> Adaptive S5->S7 decision."""

    def __init__(self, photo_ocr, bib_filter):
        self.photo_ocr = photo_ocr
        self.bib_filter = bib_filter
        self.years = tuple(str(y) for y in bib_filter.rules.get("years", ()))

    def _frame(self, reader: VideoReader, fraction: float, frames: dict) -> list[rz.FrameBib]:
        info = reader.info
        idx = frame_index(fraction * info.duration_s, info.fps, info.frame_count)
        image, pts, seek_ms, dec_ms = reader.frame_rgb(idx)
        t = time.perf_counter()
        res = self.photo_ocr.process(image)
        engine_ms = (time.perf_counter() - t) * 1000
        from ocr.bib_filter import PhotoCandidate, evaluate_photo
        cands = [PhotoCandidate(b.bib, b.numeric_identifier, b.best_confidence,
                                tuple(r.raw_text for r in res.readings if r.accepted and r.bib == b.bib),
                                "person_body", b.best_box.height, tuple(b.region_ids)) for b in res.bibs]
        bibs = []
        for b, fr in zip(res.bibs, evaluate_photo(self.bib_filter, cands)):   # V1.3.1: frame-level context
            bibs.append(rz.FrameBib(b.bib, b.numeric_identifier, round(b.best_confidence, 4), fr.accepted,
                                    tuple(fr.reasons)))
        frames[fraction] = {
            "fraction": fraction, "frame_index": idx, "pts_s": round(pts, 4), "seek_ms": round(seek_ms, 1),
            "decode_ms": round(dec_ms, 1), "engine_ms": round(engine_ms, 1),
            "persons": len(res.detection.persons),
            "bibs": [{"bib": b.bib, "numeric": b.numeric, "conf": b.confidence, "filter_accepted": b.filter_accepted,
                      "filter_reasons": list(b.filter_reasons), "height": round(c.height or 0, 1),
                      "regions": list(c.regions)} for b, c in zip(bibs, cands)],
            "readings": [{"raw": r.raw_text, "conf": round(r.ocr_confidence, 4), "accepted": r.accepted}
                         for r in res.readings][:12],
            "_image": image}
        return bibs

    def recognize(self, path: str) -> dict:
        """Result dict for video_store.save_video_result (plus the representative image)."""
        frames: dict[float, dict] = {}
        with VideoReader(path) as reader:
            s5 = [self._frame(reader, f, frames) for f in rz.S5_POSITIONS]
            decision = rz.decide_adaptive(s5, lambda: [self._frame(reader, f, frames)
                                                       for f in rz.ESCALATION_POSITIONS], self.years)
        t = decision.tally
        frames_used = decision.frames_used
        candidates = []
        for key, votes in t.ranked():
            confs = t.confs.get(key, [])
            forms = sorted(t.forms.get(key, {key}), key=lambda f: (-len(f), f))
            candidates.append({"bib_number": forms[0], "numeric_bib": key,
                               "role": "accepted" if key == decision.key else "candidate",
                               "votes": votes, "frames": frames_used,
                               "best_confidence": round(max(confs), 4) if confs else None,
                               "forms": "|".join(forms)})
        # representative frame: the best reading of the accepted bib, else the middle frame
        rep = 0.5
        if decision.key is not None:
            best = -1.0
            for f, fr in frames.items():
                for b in fr["bibs"]:
                    k = b["numeric"] or b["bib"]
                    if k == decision.key and b["filter_accepted"] and b["conf"] > best:
                        best, rep = b["conf"], f
        image = frames.get(rep, frames[0.5])["_image"]
        evidence = {"frames": [{k: v for k, v in fr.items() if k != "_image"} for _, fr in sorted(frames.items())],
                    "tally": dict(t.ranked()), "years_excluded": list(self.years),
                    "escalated": decision.escalated, "decision": decision.status,
                    "accepted": decision.key, "representative_fraction": rep}
        for fr in frames.values():
            fr.pop("_image", None)
        return {"status": decision.status, "strategy": rz.STRATEGY, "frames_used": frames_used,
                "escalated": decision.escalated, "accepted_bib": decision.full_bib(),
                "accepted_numeric": decision.key,
                "sample_times": [fr["pts_s"] for _, fr in sorted(frames.items())],
                "evidence": evidence, "candidates": candidates, "image": image}


def analyze_videos(conn: sqlite3.Connection, folder: str, event_id: int, cfg, get_recognizer: Callable,
                   photo_settings_hash: str = "", years: tuple[str, ...] = (), *,
                   progress: Callable[[int, int, str], None] | None = None,
                   thumbnail_cache=None, limit: int | None = None) -> VideoAnalyzeSummary:
    """get_recognizer() -> VideoRecognizer; called only when there is something to analyse.
    progress(i, n, path) is called BEFORE video i (videos 1 .. i-1 are committed) and may raise
    KeyboardInterrupt to cancel."""
    t0 = time.perf_counter()
    s = VideoAnalyzeSummary()
    root = os.path.abspath(folder)
    s.scan = sync_videos(conn, event_id, root, cfg)
    settings_hash = video_settings_hash(photo_settings_hash, years)
    todo, s.skipped = select_videos(conn, event_id, root, settings_hash)
    if limit:
        todo = todo[:limit]
    s.to_process = len(todo)
    if not todo:
        s.elapsed_s = time.perf_counter() - t0
        return s
    recognizer = get_recognizer()
    s.run_id = db.start_run(conn, event_id, source="video", source_folder=root,
                            settings_json=json.dumps({"strategy": rz.STRATEGY, "photo_settings": photo_settings_hash}),
                            settings_hash=settings_hash, photos_total=len(todo))
    first = None
    status = "finished"
    try:
        for i, row in enumerate(todo, start=1):
            if progress:
                progress(i, len(todo), row["file_path"])
            if first is None:
                first = time.perf_counter()
            t = time.perf_counter()
            try:
                result = recognizer.recognize(row["file_path"])
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad video must not stop the run
                vs.save_video_result(conn, row["id"], s.run_id, None, error=f"{type(exc).__name__}: {exc}"[:500],
                                     settings_hash=settings_hash, processing_ms=(time.perf_counter() - t) * 1000)
                s.failed += 1
                continue
            image = result.pop("image")
            if thumbnail_cache is not None and row["fingerprint"]:
                from utils.thumbnails import save_from_image
                if save_from_image(image, thumbnail_cache, row["fingerprint"], cfg.thumbnails, overwrite=True):
                    result["thumbnail_fingerprint"] = row["fingerprint"]
            del image
            vs.save_video_result(conn, row["id"], s.run_id, result, settings_hash=settings_hash,
                                 processing_ms=(time.perf_counter() - t) * 1000)
            s.processed += 1
            s.frames += result["frames_used"]
            s.escalated += int(result["escalated"])
            s.verified += result["status"] == rz.VERIFIED
            s.ambiguous += result["status"] == rz.AMBIGUOUS
            s.unindexed += result["status"] == rz.UNINDEXED
    except KeyboardInterrupt:
        s.interrupted, status = True, "cancelled"
    finally:
        db.finish_run(conn, s.run_id, status, s.processed)
    s.elapsed_s = time.perf_counter() - t0
    if first and s.processed:
        s.seconds_per_video = round((time.perf_counter() - first) / s.processed, 3)
    return s
