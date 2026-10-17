"""Plain data passed between the service layer and the GUI (no Qt, no engine imports)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuStatus:
    available: bool
    name: str = ""            # e.g. "NVIDIA GeForce RTX 3060"
    detail: str = ""          # reason when not available, or extra information
    cuda: str = ""            # CUDA version PaddlePaddle uses, e.g. "13.0" (7.3A)
    paddle: str = ""          # PaddlePaddle version, e.g. "3.3.1"; "" = not checked / not importable
    cudnn: str = ""


@dataclass(frozen=True)
class FolderInfo:
    folder: str
    photo_count: int
    estimated_seconds: float | None = None
    video_count: int = 0                 # V1.3: .mp4/.mov/.m4v files analysed after the photos


@dataclass(frozen=True)
class ProgressInfo:
    event: str
    total: int
    done: int
    status: str                          # "scanning", "loading models", "analysing", ...
    photos_per_second: float | None = None
    remaining_seconds: float | None = None
    current_file: str = ""
    folder: str = ""                     # 7.3A

    @property
    def percent(self) -> int:
        return int(100 * self.done / self.total) if self.total else 0


@dataclass(frozen=True)
class EventPlan:
    """What Start would do (read-only look at the index; nothing is created). 7.3A"""
    event: str
    folder: str
    exists: bool                         # the event is already in the index
    indexed_photos: int = 0              # photos of that event already in the index


@dataclass(frozen=True)
class EventSummary:
    """One event on the Dashboard (Phase 8.2; read-only counts from the index)."""
    name: str
    photos: int = 0                      # photos in the index
    analysed: int = 0                    # analysis_status 'done'
    failed: int = 0                      # analysis_status 'error'
    pending: int = 0                     # waiting for analysis (e.g. after a cancel)
    missing: int = 0                     # original file not found any more
    bib_numbers: int = 0                 # distinct searchable numbers (noise filter accepted)
    last_analysis: str = ""              # 'YYYY-MM-DD HH:MM' of the last finished/cancelled run
    folders: tuple[str, ...] = ()        # folders analysed into the event
    videos: int = 0                      # V1.3: videos in the index
    videos_verified: int = 0             # one accepted bib
    videos_ambiguous: int = 0            # several runners / conflicting evidence (not indexed)
    videos_unindexed: int = 0            # no bib reached the vote threshold
    videos_pending: int = 0
    videos_failed: int = 0


@dataclass(frozen=True)
class AnalysisSummary:
    event: str
    total: int
    processed: int
    skipped: int
    failed: int
    elapsed_seconds: float
    status: str                          # "done", "cancelled" or "error"
    message: str = ""


@dataclass(frozen=True)
class SearchResult:
    """One result card: one photo and one searched number (as in the V0.3.1 search)."""
    event: str
    bib: str                              # the searched number
    file_path: str
    thumbnail_path: str | None = None     # None: show a placeholder
    photo_id: int | None = None           # index row, needed for reject / restore
    bibs: tuple[str, ...] = ()            # full bibs shown on the card, e.g. ("B60914",)
    active_bibs: tuple[str, ...] = ()     # AI bibs that can be rejected
    rejected_bibs: tuple[str, ...] = ()   # bibs rejected by the photographer (can be restored)
    filter_reason: str = ""               # noise-filter reason if a shown bib is filtered
    missing: bool = False                 # original photo not found on disk
    match: str = "numeric"                # search tier: exact | numeric | normalized (V1.2) | alias (V1.3.1)
    matched_text: str = ""                # V1.2 normalized match: the OCR text, e.g. "60914B"
    media_type: str = "PHOTO"             # V1.3: PHOTO | VIDEO
    video_id: int | None = None           # V1.3: videos.id (reject / restore of a video bib)
    recognition_status: str = ""          # V1.3 videos: VERIFIED | AMBIGUOUS
    votes: int = 0                        # V1.3 videos: frames that read this bib
    frames: int = 0                       # V1.3 videos: frames evaluated (5 or 7)
    confirmed: bool = True                # V1.3 videos: False = ambiguous candidate, not a match
    low_confidence: str = ""              # V1.3.1: reason if the shown bib is an uncertain candidate

    @property
    def is_video(self) -> bool:
        return self.media_type == "VIDEO"

    @property
    def file_name(self) -> str:
        return self.file_path.replace("/", "\\").rsplit("\\", 1)[-1]

    @property
    def filtered(self) -> bool:
        return bool(self.filter_reason)


def format_duration(seconds: float | None) -> str:
    """'1 h 33 min', '12 min', '45 s'; '-' if unknown."""
    if seconds is None:
        return "-"
    seconds = max(0, int(round(seconds)))
    if seconds >= 3600:
        return f"{seconds // 3600} h {seconds % 3600 // 60} min"
    if seconds >= 60:
        return f"{seconds // 60} min"
    return f"{seconds} s"
