"""
Phase 7.3A/7.3B: the real analysis service layer, with the analysis in a separate
worker process (services/analysis_worker.py, started through AnalysisJob).

Search / events / thumbnails / reject-restore: unchanged from Phase 7.2
(RunnerFinderSearchService). Analysis side:

    check_gpu()            worker "gpu-check": utils.gpu_runtime.gpu_status() in the
                           GPU environment; PaddlePaddle is imported THERE, never in
                           the GUI; no OCR model is loaded
    inspect_folder(path)   photo count with the engine's discovery rules
                           (scanner.photo_scanner.iter_photo_files), background thread
    estimate_duration(n)   0.8 s per photo (RTX 3060, measured on a real event)
    prepare_event(e, f)    read-only look at the index: new event, or continue an existing one
    start_analysis(e, f)   worker "analyse" (7.3B): the unchanged V0.3.1 analysis
                           (indexing.analyze.analyze_folder, Lite Top 1, OCR on the GPU)
                           into this service's index and thumbnail cache; incremental,
                           one commit per photo, resumable
    cancel_analysis()      "cancel" to the worker: it stops after the current photo (saved);
                           killed only if it does not stop within CANCEL_GRACE_MS
"""
from __future__ import annotations

import os
import sqlite3
import time

from PySide6.QtCore import QObject

from desktop.services import progress_protocol as proto
from desktop.services.analysis_job import AnalysisJob
from desktop.services.demo_service import SECONDS_PER_PHOTO_GPU
from desktop.services.models import AnalysisSummary, EventPlan, FolderInfo, GpuStatus, ProgressInfo
from desktop.services.runnerfinder_service import RunnerFinderSearchService


#: V1.3: measured S5 mean on the 30 evidence clips (RTX 3060), about 3.2 s per video
SECONDS_PER_VIDEO_GPU = 3.2


def count_videos(folder: str) -> int:
    """Videos the analysis would find (V1.3: .mp4 .mov .m4v, same discovery rules). Read-only."""
    from config import get_config
    from scanner.photo_scanner import iter_photo_files
    cfg = get_config()
    return sum(1 for _ in iter_photo_files(folder, (".mp4", ".mov", ".m4v"), cfg.scan.recursive,
                                           cfg.scan.skip_dir_names,
                                           (cfg.paths.default_export_root, cfg.paths.data_dir)))


def count_photos(folder: str) -> int:
    """Photos the real analysis would find: the engine's discovery rules. Read-only."""
    from config import get_config
    from scanner.photo_scanner import iter_photo_files
    cfg = get_config()
    return sum(1 for _ in iter_photo_files(folder, cfg.scan.extensions, cfg.scan.recursive,
                                           cfg.scan.skip_dir_names,
                                           (cfg.paths.default_export_root, cfg.paths.data_dir)))


#: a photo takes about 1 s; model loading (not interruptible) about 5-15 s
CANCEL_GRACE_MS = 60000


class RealAnalysisService(RunnerFinderSearchService):
    """Real search (7.2) + real analysis in a worker process (7.3B)."""

    def __init__(self, parent: QObject | None = None, db_path=None, thumbnail_cache=None,
                 worker_python: str | None = None, simulate_seconds: float | None = None):
        super().__init__(parent, db_path=db_path, thumbnail_cache=thumbnail_cache)
        self.mode_description = (f"Index: {self.db_path}  ·  Analysis: RunnerFinder V0.3.1 engine in a "
                                 "worker process (GPU, Lite Top 1)")
        self.worker_python = worker_python
        self.simulate_seconds = simulate_seconds      # tests only: Phase 7.3A dry run instead of OCR
        self._gpu_job: AnalysisJob | None = None
        self._job: AnalysisJob | None = None
        self._run: dict | None = None
        self.last_result: dict = {}

    # ------------------------------------------------------------------ state
    @property
    def connected(self) -> bool:
        return True                           # real worker framework: Start requires the GPU

    @property
    def busy(self) -> bool:
        return self._job is not None

    # ------------------------------------------------------------------ GPU
    def check_gpu(self) -> None:
        if self._gpu_job is not None:
            return                            # a check is already running
        job = AnalysisJob(self, self.worker_python)
        answer: list[GpuStatus] = []

        def message(msg):
            if msg["status"] == proto.GPU:
                answer.append(GpuStatus(bool(msg.get("available")), msg.get("name", ""),
                                        msg.get("detail", ""), msg.get("cuda", ""),
                                        msg.get("paddle", ""), msg.get("cudnn", "")))

        def finished(code, stderr):
            self._gpu_job = None
            job.deleteLater()
            if not answer:
                last = stderr.strip().splitlines()[-1] if stderr.strip() else f"worker exit code {code}"
                answer.append(GpuStatus(False, "", f"GPU check failed: {last}"))
            self.gpu_status_changed.emit(answer[0])
        job.message.connect(message)
        job.finished.connect(finished)
        self._gpu_job = job
        job.start(["gpu-check"])

    def probe_gpu(self) -> None:              # 7.1 name
        self.check_gpu()

    # ------------------------------------------------------------------ folder / estimate / event
    def estimate_duration(self, photo_count: int) -> float | None:
        return photo_count * SECONDS_PER_PHOTO_GPU if photo_count else None

    def inspect_folder(self, folder: str) -> None:
        if not folder or not os.path.isdir(folder):
            self.error.emit(f"Folder not found: {folder}")
            return
        def counts():
            return count_photos(folder), count_videos(folder)

        def done(nv):
            n, v = nv
            est = (self.estimate_duration(n) or 0) + v * SECONDS_PER_VIDEO_GPU
            self.folder_inspected.emit(FolderInfo(folder, n, est or None, v))
        self._counter.run(counts,                                # the demo service's counting thread
                          done,
                          lambda message: self.error.emit(f"Cannot read {folder}: {message}"))

    def prepare_event(self, event: str, folder: str) -> EventPlan:
        """Read-only (the index is opened with mode=ro): nothing is created in 7.3A."""
        conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT id FROM events WHERE name = ?", (event,)).fetchone()
            n = conn.execute("SELECT COUNT(*) FROM photos WHERE event_id = ?", (row[0],)).fetchone()[0] if row else 0
        finally:
            conn.close()
        return EventPlan(event, folder, row is not None, n)

    # ------------------------------------------------------------------ analysis
    def start_analysis(self, event: str, folder: str) -> None:
        event, folder = (event or "").strip(), (folder or "").strip()
        if self._job is not None:
            self.error.emit("An analysis is already running.")
            return
        if not event:
            self.error.emit("Please enter an event name.")
            return
        if not os.path.isdir(folder):
            self.error.emit(f"Folder not found: {folder}")
            return
        try:
            plan = self.prepare_event(event, folder)
        except sqlite3.Error as exc:
            self.error.emit(f"Cannot read the index: {exc}")
            return
        note = (f"existing event, {plan.indexed_photos:,} photos indexed: only new, changed and failed "
                "photos are analysed" if plan.exists else "new event")
        self._run = {"event": event, "folder": folder, "total": 0, "done": 0, "start": time.monotonic(),
                     "note": note}
        job = AnalysisJob(self, self.worker_python, cancel_grace_ms=CANCEL_GRACE_MS)
        job.message.connect(self._worker_message)
        job.finished.connect(self._worker_finished)
        self._job = job
        self.analysis_progress.emit(ProgressInfo(event, 0, 0, f"Starting… ({note})", folder=folder))
        args = ["analyse", "--folder", folder, "--event", event,
                "--db", str(self.db_path), "--thumbnail-cache", str(self.thumbnail_cache)]
        if self.simulate_seconds is not None:
            args += ["--simulate", str(self.simulate_seconds)]
        job.start(args)

    def cancel_analysis(self) -> None:
        if self._job is not None:
            self._job.cancel()
            r = self._run
            self.analysis_progress.emit(ProgressInfo(
                r["event"], r["total"], r["done"], "Cancelling… (the current photo is finished and saved)",
                current_file=r.get("file", ""), folder=r["folder"]))

    def shutdown(self, msecs: int = 20000) -> None:
        """Window closing: stop the worker (it would also stop on its own when stdin closes)."""
        for job in (self._job, self._gpu_job):
            if job is not None and job.running:
                job.cancel()
                if not job.wait(msecs):
                    job.kill()
        self.wait_idle(msecs)

    def _worker_message(self, msg: dict) -> None:
        r = self._run
        if r is None:
            return
        st = msg["status"]
        if st == proto.SCANNING:
            found = int(msg.get("found") or 0)
            text = (f"Scanning folder… {found:,} photos found" if not msg.get("total")
                    else f"Scan complete: {int(msg['total']):,} photos in the folder")
            self.analysis_progress.emit(ProgressInfo(r["event"], 0, 0, text, folder=r["folder"]))
        elif st == proto.LOADING_MODELS:
            self.analysis_progress.emit(ProgressInfo(
                r["event"], 0, 0, "Loading models (YOLO person detector + PaddleOCR on the GPU)…",
                folder=r["folder"]))
        elif st == proto.VIDEO_ANALYSING:                   # V1.3: video stage after the photos
            r["total"], r["done"], r["file"] = int(msg["total"]), int(msg.get("completed", 0)), msg.get("filename", "")
            per_video = msg.get("speed")
            self.analysis_progress.emit(ProgressInfo(
                r["event"], r["total"], r["done"], f"Analysing videos ({r['total']:,} to analyse, 5-7 frames each)",
                1 / per_video if per_video else None, msg.get("remaining"), r["file"], folder=r["folder"]))
        elif st == proto.ANALYSING:
            r["total"], r["done"], r["file"] = int(msg["total"]), int(msg.get("completed", 0)), msg.get("filename", "")
            per_photo = msg.get("speed")
            self.analysis_progress.emit(ProgressInfo(
                r["event"], r["total"], r["done"], f"Analysing ({r['note']})",
                1 / per_photo if per_photo else None, msg.get("remaining"), r["file"], folder=r["folder"]))

    def _worker_finished(self, code: int, stderr: str) -> None:
        job, self._job = self._job, None
        r, self._run = self._run, None
        final = job.final if job is not None else None
        if job is not None:
            job.deleteLater()
        if r is None:
            return
        final = final or {}
        elapsed = float(final.get("duration") or (time.monotonic() - r["start"]))
        skipped = int(final.get("skipped") or 0)
        vtotal = int(final.get("video_total") or 0)
        video_text = ""
        if vtotal or final.get("video_skipped"):           # V1.3
            video_text = (f" Videos: {int(final.get('video_processed') or 0):,} analysed "
                          f"({int(final.get('video_verified') or 0):,} with a bib, "
                          f"{int(final.get('video_ambiguous') or 0):,} ambiguous, "
                          f"{int(final.get('video_unindexed') or 0):,} not indexed"
                          + (f", {int(final.get('video_failed') or 0):,} unreadable" if final.get("video_failed") else "")
                          + ")"
                          + (f", {int(final.get('video_skipped') or 0):,} already analysed" if final.get("video_skipped") else "")
                          + ".")
        if final.get("status") == proto.FINISHED:
            status = "done"
            if not final.get("total") and not vtotal:
                msg = (f"Nothing to analyse: all {skipped:,} photos are already in the index." + video_text)
            else:
                msg = (f"Analysis completed. Thumbnails: {int(final.get('thumbnails') or 0):,}."
                       + (f" {skipped:,} photos were already analysed and skipped." if skipped else "") + video_text)
        elif final.get("status") == proto.CANCELLED:
            status, msg = "cancelled", ("Cancelled. The finished photos are saved; start again with the same "
                                        "folder and event to continue where it stopped.")
        else:
            status = "error"
            detail = final.get("message") or (stderr.strip().splitlines()[-1] if stderr.strip()
                                                       else f"worker stopped (exit code {code})")
            msg = f"The analysis worker stopped: {detail}"
        processed = int(final.get("processed", r["done"]))
        total = int(final.get("total", r["total"]))
        failed = int(final.get("failed") or 0)
        self.last_result = final                       # full worker summary (tests, report)
        self.analysis_finished.emit(AnalysisSummary(r["event"], total, processed, skipped, failed, elapsed,
                                                    status, msg))
