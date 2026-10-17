"""
Demo implementation of EngineService for the Phase 7.1 skeleton.

* No RunnerFinder engine module is imported; nothing is written anywhere.
* inspect_folder: really counts .jpg/.jpeg/.png files (read-only, background thread).
* check_gpu: reports "engine not connected".
* start_analysis: a SIMULATED run driven by a timer, so the progress page
  and its signals can be exercised. No photo is opened, no OCR runs.
* search: returns no results ("engine not connected").
"""
from __future__ import annotations

import os
import time

from PySide6.QtCore import QObject, QTimer

from desktop.services.background import BackgroundRunner
from desktop.services.engine_service import EngineService
from desktop.services.models import AnalysisSummary, EventSummary, FolderInfo, GpuStatus, ProgressInfo

PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png")
#: measured on a real event with an RTX 3060 (V0.3.1 validation)
SECONDS_PER_PHOTO_GPU = 0.8


def estimate_seconds(photo_count: int) -> float | None:
    return photo_count * SECONDS_PER_PHOTO_GPU if photo_count else None


def count_photos(folder: str) -> int:
    """Photos in folder and sub-folders (hidden folders skipped). Read-only."""
    n = 0
    for d, dirs, files in os.walk(folder):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        n += sum(1 for f in files if f.lower().endswith(PHOTO_EXTENSIONS))
    return n


class DemoEngineService(EngineService):
    mode_description = "Demo mode — RunnerFinder engine not connected (analysis is simulated, no OCR)"

    def __init__(self, parent: QObject | None = None, tick_ms: int = 40, photos_per_tick: int = 7):
        super().__init__(parent)
        self._counter = BackgroundRunner(self)
        self._timer = QTimer(self)
        self._timer.setInterval(tick_ms)
        self._timer.timeout.connect(self._tick)
        self._photos_per_tick = photos_per_tick
        self._run: dict | None = None
        self._last: dict | None = None              # the last simulated run (Dashboard)

    # -------------------------------------------------------------- state
    @property
    def busy(self) -> bool:
        return self._run is not None

    # -------------------------------------------------------------- GPU / folder / events
    def check_gpu(self) -> None:
        self.gpu_status_changed.emit(GpuStatus(False, "", "not checked: RunnerFinder engine not connected"))

    def inspect_folder(self, folder: str) -> None:
        if not folder or not os.path.isdir(folder):
            self.error.emit(f"Folder not found: {folder}")
            return
        def count():
            n = count_photos(folder)
            return FolderInfo(folder, n, estimate_seconds(n))
        self._counter.run(count, self.folder_inspected.emit,
                          lambda message: self.error.emit(f"Cannot read {folder}: {message}"))

    def estimate_duration(self, photo_count: int) -> float | None:
        return estimate_seconds(photo_count)

    def list_events(self) -> None:
        names = [self._run["event"]] if self._run else []
        self.events_changed.emit(names)

    def list_event_summaries(self) -> None:
        r = self._last or self._run
        self.event_summaries_changed.emit([EventSummary(r["event"], r["total"], r["done"])] if r else [])

    def search(self, event: str, bib: str, include_rejected: bool = False,
               include_noise: bool = False, include_ambiguous: bool = False) -> None:
        self.search_finished.emit(event, bib, [])

    def reject(self, result) -> None:
        self.error.emit("Review is not available: RunnerFinder engine not connected.")

    def restore(self, result) -> None:
        self.error.emit("Review is not available: RunnerFinder engine not connected.")

    # -------------------------------------------------------------- simulated analysis
    def start_analysis(self, event: str, folder: str, total: int | None = None) -> None:
        if self._run is not None:
            self.error.emit("An analysis is already running.")
            return
        total = count_photos(folder) if total is None else total
        self._run = {"event": event, "total": total, "done": 0, "start": time.monotonic()}
        self.analysis_progress.emit(ProgressInfo(event, total, 0, "starting (simulated)"))
        self._timer.start()

    def cancel_analysis(self) -> None:
        if self._run is not None:
            self._finish("cancelled", "Cancelled by the user (simulated). A real run continues where it stopped.")

    def _tick(self) -> None:
        r = self._run
        if r is None:
            self._timer.stop()
            return
        r["done"] = min(r["total"], r["done"] + self._photos_per_tick)
        elapsed = max(1e-6, time.monotonic() - r["start"])
        speed = r["done"] / elapsed
        remaining = (r["total"] - r["done"]) / speed if speed > 0 else None
        self.analysis_progress.emit(ProgressInfo(
            r["event"], r["total"], r["done"], "analysing (simulated)", speed, remaining,
            f"photo {r['done']} of {r['total']}"))
        if r["done"] >= r["total"]:
            self._finish("done", "Simulated analysis finished. No photo was read.")

    def _finish(self, status: str, message: str) -> None:
        self._timer.stop()
        r, self._run = self._run, None
        self._last = r
        self.analysis_finished.emit(AnalysisSummary(
            r["event"], r["total"], r["done"], 0, 0, time.monotonic() - r["start"], status, message))
