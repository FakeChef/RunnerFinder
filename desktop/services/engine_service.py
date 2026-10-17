"""
The only interface between the desktop GUI and RunnerFinder.

The GUI calls the methods; every answer comes back as a Qt signal, so slow
work (folder counting, GPU probing, a 1.5-hour analysis) never blocks the
window. Phase 7.1: DemoEngineService. Phase 7.2: RunnerFinderSearchService
(real search, in-process). Phase 7.3A: RealAnalysisService (analysis in a
separate worker process; dry run, no OCR yet).
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class EngineService(QObject):
    # answers (all payloads are dataclasses from services/models.py)
    gpu_status_changed = Signal(object)        # GpuStatus
    folder_inspected = Signal(object)          # FolderInfo
    analysis_progress = Signal(object)         # ProgressInfo
    analysis_finished = Signal(object)         # AnalysisSummary
    events_changed = Signal(list)              # list[str] of event names
    event_summaries_changed = Signal(list)     # list[EventSummary] (Dashboard, Phase 8.2)
    search_finished = Signal(str, str, list)   # event, bib, list[SearchResult]
    search_hidden = Signal(int)                # candidates hidden by the noise filter (last search)
    review_finished = Signal(str)              # reject/restore stored; message for the user
    error = Signal(str)                        # message for the user

    #: shown in the window when the real engine is not connected
    mode_description = ""

    @property
    def connected(self) -> bool:
        """True when the ANALYSIS is backed by the real RunnerFinder engine."""
        return False

    @property
    def search_connected(self) -> bool:
        """True when events / search / review use the real RunnerFinder index."""
        return False

    @property
    def busy(self) -> bool:
        """True while an analysis is running."""
        return False

    def check_gpu(self) -> None:
        """Answer: gpu_status_changed(GpuStatus). Environment check only, no OCR model."""
        raise NotImplementedError

    def inspect_folder(self, folder: str) -> None:
        """Answer: folder_inspected(FolderInfo) with the photo count and the estimate."""
        raise NotImplementedError

    def estimate_duration(self, photo_count: int) -> float | None:
        """Estimated analysis time in seconds (synchronous, no I/O)."""
        raise NotImplementedError

    def start_analysis(self, event: str, folder: str) -> None:
        """Answers: analysis_progress(ProgressInfo)..., then analysis_finished(AnalysisSummary)."""
        raise NotImplementedError

    def cancel_analysis(self) -> None:
        raise NotImplementedError

    def list_events(self) -> None:
        raise NotImplementedError

    def list_event_summaries(self) -> None:
        """Answer: event_summaries_changed(list[EventSummary]) - read-only (Dashboard)."""
        raise NotImplementedError

    def search(self, event: str, bib: str, include_rejected: bool = False,
               include_noise: bool = False, include_ambiguous: bool = False) -> None:
        raise NotImplementedError

    def reject(self, result) -> None:
        """Hide the card's AI bibs for this photo (photographer verdict 'incorrect')."""
        raise NotImplementedError

    def restore(self, result) -> None:
        """Undo a reject for this photo."""
        raise NotImplementedError
