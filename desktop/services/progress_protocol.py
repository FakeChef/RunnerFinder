"""
Messages between the analysis worker process and the GUI (Phase 7.3A / 7.3B).

One JSON object per line on the worker's stdout; every message has "status".
Standard library only: the worker runs in another Python environment
(the GPU engine environment) that has no Qt.

Worker -> GUI
    {"status": "gpu", "available": true, "name": "NVIDIA GeForce RTX 3060",
     "cuda": "13.0", "cudnn": "9.13", "paddle": "3.3.1", "detail": ""}
    {"status": "scanning", "total": 0, "found": 2400, ...}       while the folder is scanned
    {"status": "scanning", "total": 6974, "found": 6974, ...}    scan complete
    {"status": "loading_models", "mode": "lite1", "ocr_device": "gpu"}
    {"status": "models_loaded", "ocr_device": "gpu:0", "gpu": "NVIDIA GeForce RTX 3060", "cuda": "13.0"}
    {"status": "analysing", "current": 1234, "completed": 1233, "total": 6974,
     "filename": "IMG_1234.JPG", "speed": 0.78, "remaining": 4477.7}
        current = photo being analysed; speed = seconds per photo (null for the first photo)
    {"status": "video_analysing", "current": 3, "completed": 2, "total": 53, "filename": "A_0047C001.MP4",
     "speed": 3.1, "remaining": 158.1}      V1.3: the video stage, after the photos (speed = s/video)
    {"status": "finished", "processed": 6974, "failed": 0, "duration": 5560.3, "total": 6974,
     "skipped": 0, "thumbnails": 6974, "run_id": 7, "settings_hash": "...", "ram_peak_mib": ...}
    {"status": "cancelled", ...same fields as finished...}
    {"status": "error", "message": "Folder not found: ..."}
    V1.3 final messages add: pipeline, pipeline_fallback, kmp_blocktime and the video_* fields
    (video_total, video_processed, video_failed, video_verified, video_ambiguous, video_unindexed,
    video_escalated, video_frames, video_skipped, videos_found, seconds_per_video).

GUI -> worker (stdin, one word per line)
    cancel           stop after the current photo; closing stdin does the same
"""
from __future__ import annotations

import json

GPU, SCANNING, ANALYSING, FINISHED, CANCELLED, ERROR = (
    "gpu", "scanning", "analysing", "finished", "cancelled", "error")
LOADING_MODELS, MODELS_LOADED = "loading_models", "models_loaded"
VIDEO_ANALYSING = "video_analysing"
FINAL = (FINISHED, CANCELLED, ERROR)
CANCEL_COMMAND = "cancel"


def encode(status: str, **fields) -> str:
    """One protocol line (without the newline)."""
    return json.dumps({"status": status, **fields}, ensure_ascii=False)


def decode(line: str | bytes) -> dict | None:
    """The message in a line, or None for anything else (library output, warnings)."""
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        msg = json.loads(line)
    except ValueError:
        return None
    return msg if isinstance(msg, dict) and isinstance(msg.get("status"), str) else None
