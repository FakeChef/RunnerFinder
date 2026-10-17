"""
Analysis worker: runs in its OWN process, never inside the GUI (Phase 7.3B).

    <python> -m desktop.services.analysis_worker gpu-check
    <python> -m desktop.services.analysis_worker analyse --folder D:\\RacePhotos\\MyRace2026 --event "My Race 2026"
             [--db data\\runner_photo_finder.db] [--thumbnail-cache data\\thumbnails] [--limit N]

<python> is the GPU engine environment (see services/analysis_job.py); the working
directory is the repository folder. Messages go to stdout as JSON lines
(services/progress_protocol.py); "cancel" on stdin, or stdin closing because
the GUI went away, stops the run after the current photo.

    gpu-check  utils.gpu_runtime.gpu_status(): imports PaddlePaddle and runs one
               tiny tensor on the GPU. Environment check only, no OCR model.
    analyse    the UNCHANGED V0.3.1 analysis, exactly as
                   index_cli.py analyze <folder> --event <event> --ocr-device gpu
               (what the RC1 release runs): config.get_config() with
               processing_mode "lite1" (Lite Top 1) and ocr.device "gpu",
               database.init_db(), the CLI's model loading (YOLO person detector +
               PaddleOCR), then indexing.analyze.analyze_folder(). Incremental and
               resumable (Phase 4), one commit per photo, thumbnails during the
               analysis, bib noise filter: all done by the engine itself.

Cancel = the engine's own Ctrl+C path: the progress callback, which the
engine calls BEFORE each photo (the previous photo is already committed),
raises KeyboardInterrupt; analyze_folder() catches it, the run is stored as
'cancelled' and the remaining photos stay 'pending' for the next run.

--simulate SECONDS (tests only, the application never passes it): the
Phase 7.3A dry run (scan, count, a delay per photo; nothing written).

Standard library + unchanged engine modules only: no Qt in this process.
Everything the engine or its libraries print goes to stderr; stdout carries
only protocol lines.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

from desktop.services import progress_protocol as proto

#: at most this many progress lines per second (a 7000-photo run must not flood the GUI)
MAX_MESSAGES_PER_SECOND = 10
#: V1.2: photos decoded ahead of the analysis by a producer thread (0 = off, the V1.1 sequence).
#: The environment variable RUNNERFINDER_PREFETCH overrides it (troubleshooting).
DEFAULT_PREFETCH = 2
#: V1.3: "s3" = a second thread also runs the person detection ahead (scanner/pipeline3.py);
#: "prefetch" = the V1.2 pipeline. The environment variable RUNNERFINDER_PIPELINE overrides it.
DEFAULT_PIPELINE = "s3"
PIPELINES = ("s3", "prefetch")
#: the analysis settings validated in Phase 3/4 and used by the RC1 release
PROCESSING_MODE = "lite1"
OCR_DEVICE = "gpu"

_protocol_out = None                      # set by main(): the original stdout


def send(status: str, **fields) -> None:
    out = _protocol_out or sys.stdout
    out.write(proto.encode(status, **fields) + "\n")
    out.flush()


def _protect_protocol_stream() -> None:
    """Protocol lines keep the real stdout; print()s of the engine and C/C++ libraries
    (Paddle, YOLO) are redirected to stderr, so they can never break a JSON line."""
    global _protocol_out
    try:
        fd = os.dup(1)
        os.dup2(2, 1)                       # C-level stdout -> stderr
        _protocol_out = open(fd, "w", encoding="utf-8", errors="replace", buffering=1, closefd=True)
        sys.stdout = sys.stderr             # Python-level print() -> stderr
    except OSError:
        _protocol_out = sys.stdout


class CancelWatcher:
    """Watches stdin in the background: 'cancel' or end of input (the GUI is gone) sets the flag.

    Windows: the pipe is POLLED with PeekNamedPipe and only read when data is
    waiting. A thread blocked in a read on stdin deadlocks DLL initialisation
    that touches the standard handles (numpy/cv2 loading for YOLO hung forever
    in Phase 7.3B testing)."""

    POLL_SECONDS = 0.2

    def __init__(self, stream=None):
        self.cancelled = threading.Event()
        self._stream = stream if stream is not None else sys.stdin
        target = self._poll_windows_pipe if sys.platform == "win32" and stream is None else self._read
        threading.Thread(target=target, daemon=True).start()

    def _read(self) -> None:
        try:
            for line in self._stream:
                if line.strip().lower() == proto.CANCEL_COMMAND:
                    break
        except (OSError, ValueError):
            pass
        self.cancelled.set()                 # 'cancel', or stdin closed: the GUI is gone

    def _poll_windows_pipe(self) -> None:
        import ctypes
        import msvcrt
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.PeekNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p,
                                           ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        try:
            handle = msvcrt.get_osfhandle(self._stream.fileno())
        except (OSError, ValueError, AttributeError):
            return                            # no stdin: nothing to watch
        if kernel32.GetFileType(wintypes.HANDLE(handle)) != 3:     # FILE_TYPE_PIPE
            return                            # console or NUL (manual runs): Ctrl+C still works
        buffer = b""
        while True:
            avail = wintypes.DWORD(0)
            if not kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(avail), None):
                self.cancelled.set()          # broken pipe: the GUI closed stdin or is gone
                return
            if avail.value:
                buffer += os.read(self._stream.fileno(), avail.value)   # data is waiting: no blocking
                *lines, buffer = buffer.split(b"\n")
                if any(x.strip().lower() == proto.CANCEL_COMMAND.encode() for x in lines):
                    self.cancelled.set()
                    return
            time.sleep(self.POLL_SECONDS)


def peak_memory() -> dict:
    """Peak RAM of this process and, if Paddle is loaded, its peak GPU memory (MiB)."""
    out = {}
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            c = Counters()
            c.cb = ctypes.sizeof(c)
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb):
                out["ram_peak_mib"] = round(c.PeakWorkingSetSize / 2**20)
        except Exception:  # noqa: BLE001 - statistics only
            pass
    if "paddle" in sys.modules:
        try:
            cuda = sys.modules["paddle"].device.cuda
            out["gpu_peak_allocated_mib"] = round(cuda.max_memory_allocated() / 2**20)
            out["gpu_peak_reserved_mib"] = round(cuda.max_memory_reserved() / 2**20)
        except Exception:  # noqa: BLE001
            pass
    return out


# ---------------------------------------------------------------------- commands
def gpu_check() -> int:
    """Environment validation with the existing engine check; no OCR model is loaded."""
    try:
        import config  # noqa: F401  - registers the CUDA/cuDNN wheel DLLs (utils.gpu_runtime)
        from utils.gpu_runtime import GPUUnavailable, gpu_status
    except Exception as exc:  # noqa: BLE001
        send(proto.GPU, available=False, detail=f"{type(exc).__name__}: {exc}")
        return 1
    try:
        info = gpu_status(0)
    except GPUUnavailable as exc:
        send(proto.GPU, available=False, detail=str(exc), python=sys.executable)
        return 1
    send(proto.GPU, available=True, name=info.get("name", "?"), cuda=str(info.get("cuda", "")),
         cudnn=str(info.get("cudnn", "")), paddle=str(info.get("paddle", "")),
         memory_mib=info.get("memory_mib"), python=sys.executable, detail="")
    return 0


def analyse(folder: str, event: str, db_path: str | None = None, thumbnail_cache: str | None = None,
            limit: int | None = None, report_modules: bool = False,
            watcher: CancelWatcher | None = None, prefetch: int = DEFAULT_PREFETCH,
            pipeline: str = DEFAULT_PIPELINE, videos: bool = True) -> int:
    """The V0.3.1 analysis (index_cli.py analyze --ocr-device gpu) with progress messages."""
    watcher = watcher or CancelWatcher()
    t0 = time.perf_counter()
    if not os.path.isdir(folder):
        send(proto.ERROR, message=f"Folder not found: {folder}")
        return 1

    from pathlib import Path

    from config import get_config
    from database.database import init_db
    from database.results_store import ResultConflict
    from indexing.analyze import analyze_folder
    from utils.gpu_runtime import GPUUnavailable

    cfg = get_config()                                   # same as index_cli.cmd_analyze
    cfg.detection.processing_mode = PROCESSING_MODE
    cfg.ocr.device = OCR_DEVICE
    if thumbnail_cache:
        cfg.paths.thumbnail_cache = Path(thumbnail_cache)   # the cache the Search page reads
    cfg.validate()
    db_path = db_path or cfg.paths.database_file
    conn = init_db(db_path)

    state = {"found": 0, "first": None, "last_msg": 0.0, "total": 0}

    def stop_if_cancelled():
        if watcher.cancelled.is_set():
            raise KeyboardInterrupt              # the engine's own interruption path

    def scan_progress(done: int, path: str) -> None:
        state["found"] = done
        now = time.perf_counter()
        if now - state["last_msg"] >= 1 / MAX_MESSAGES_PER_SECOND:
            state["last_msg"] = now
            send(proto.SCANNING, total=0, found=done, folder=folder, event=event)
        stop_if_cancelled()

    models: dict = {}

    def load_models():
        """Exactly index_cli.cmd_analyze's load_models (a local function there, so not importable).
        V1.3: loaded once per run and shared by the photo and the video stage."""
        if "pair" in models:
            return models["pair"]
        send(proto.SCANNING, total=state["found"], found=state["found"], folder=folder, event=event)
        send(proto.LOADING_MODELS, mode=PROCESSING_MODE, ocr_device=OCR_DEVICE)
        from detection.person_detector import YoloPersonDetector
        from ocr.bib_reader import PaddleOCREngine
        d = cfg.detection
        detector = None
        if PROCESSING_MODE != "center":
            detector = YoloPersonDetector(cfg.paths.models_dir / d.model_name, d.person_conf_threshold,
                                          d.detection_image_size, d.device)
        engine = PaddleOCREngine(cfg.ocr)
        send(proto.MODELS_LOADED, ocr_device=str(engine.device),
             gpu=(engine.device_info or {}).get("name", ""), cuda=str((engine.device_info or {}).get("cuda", "")),
             note=engine.device_note or "")
        models["pair"] = (detector, engine)
        return detector, engine

    def progress(i: int, n: int, path: str) -> None:
        """Called by the engine BEFORE photo i; photos 1 .. i-1 are committed."""
        stop_if_cancelled()
        now = time.perf_counter()
        state["total"] = n
        if state["first"] is None:
            state["first"] = now
        completed = i - 1
        if i == 1 or i == n or now - state["last_msg"] >= 1 / MAX_MESSAGES_PER_SECOND:
            state["last_msg"] = now
            per_photo = (now - state["first"]) / completed if completed else None
            send(proto.ANALYSING, current=i, completed=completed, total=n, filename=os.path.basename(path),
                 speed=round(per_photo, 3) if per_photo else None,
                 remaining=round(per_photo * (n - completed), 1) if per_photo else None)

    try:
        s = analyze_folder(conn, folder, event, cfg, load_models, limit=limit, debug=False,
                           progress=progress, scan_progress=scan_progress, prefetch=prefetch,
                           three_stage=(pipeline == "s3" and prefetch > 0))
    except KeyboardInterrupt:                    # cancelled while scanning (before any photo)
        conn.close()
        send(proto.CANCELLED, processed=0, failed=0, total=0, duration=round(time.perf_counter() - t0, 1),
             stage="scanning")
        return 0
    except (FileNotFoundError, NotADirectoryError, ValueError, ResultConflict) as exc:
        conn.close()
        send(proto.ERROR, message=str(exc))
        return 2
    except GPUUnavailable as exc:
        conn.close()
        send(proto.ERROR, message=f"The GPU cannot be used for OCR: {exc}")
        return 4
    photo_end = time.perf_counter()              # photo speed must not include the video stage
    video = None
    if videos and not s.interrupted:
        try:
            video = _analyse_videos(conn, folder, event, s.event_id, cfg, load_models, stop_if_cancelled,
                                    thumbnail_cache)
        except GPUUnavailable as exc:
            conn.close()
            send(proto.ERROR, message=f"The GPU cannot be used for OCR: {exc}")
            return 4
    conn.close()

    sc, sel = s.scan, s.selection
    requested = pipeline if prefetch > 0 else "sequential"
    result = dict(
        prefetch=prefetch, pipeline=s.pipeline_used or ("none" if not s.to_process else requested),
        pipeline_requested=requested,
        pipeline_fallback=s.pipeline_fallback, kmp_blocktime=os.environ.get("KMP_BLOCKTIME"),
        processed=s.processed, failed=s.failed_now, total=s.to_process, duration=round(s.elapsed_s, 1),
        skipped=sel.done if sel else 0, scanned=sc.found if sc else 0,
        new=len(sel.new) if sel else 0, changed=len(sel.changed) if sel else 0,
        retried=len(sel.failed) if sel else 0, unreadable=sc.unreadable if sc else 0,
        missing=sel.missing if sel else 0, thumbnails=s.thumbnails, run_id=s.run_id, device=s.device,
        settings_hash=s.settings_hash, recovered_runs=s.stale_runs_cancelled, event_id=s.event_id,
        run_dir=str(s.run_dir or ""), db=str(db_path),
        seconds_per_photo=round((photo_end - state["first"]) / s.processed, 3)
        if state["first"] and s.processed else None,
        **peak_memory())
    if video is not None:
        v = video
        result.update(video_total=v.to_process, video_processed=v.processed, video_failed=v.failed,
                      video_skipped=v.skipped, video_verified=v.verified, video_ambiguous=v.ambiguous,
                      video_unindexed=v.unindexed, video_escalated=v.escalated, video_frames=v.frames,
                      videos_found=v.scan.found, videos_unreadable=v.scan.unreadable,
                      videos_missing=v.scan.newly_missing, seconds_per_video=v.seconds_per_video,
                      video_duration=round(v.elapsed_s, 1), video_run_id=v.run_id)
        result.update(peak_memory())
    interrupted = s.interrupted or (video is not None and video.interrupted)
    if report_modules:
        result["modules"] = sorted({m.split(".")[0] for m in sys.modules})
    send(proto.CANCELLED if interrupted else proto.FINISHED, **result)
    return 0


def _analyse_videos(conn, folder, event, event_id, cfg, load_models, stop_if_cancelled, thumbnail_cache):
    """V1.3 video stage: the folder's videos into the same event, after the photos (video/analyze.py)."""
    from dataclasses import asdict

    from ocr.bib_filter import BibCandidateFilter
    from video.analyze import VideoRecognizer, analyze_videos
    import hashlib
    import json

    bib_filter = BibCandidateFilter.load(cfg.paths.bib_rules_file)
    years = tuple(str(y) for y in bib_filter.rules.get("years", ()))
    photo_settings = hashlib.sha256(json.dumps(
        {"detection": asdict(cfg.detection), "ocr": asdict(cfg.ocr), "bib": asdict(cfg.bib),
         "rules": bib_filter.rules}, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def get_recognizer():
        from detection.bib_detector import RegionFinder
        from ocr.photo_ocr import PhotoOCR
        detector, engine = load_models()
        d = cfg.detection
        return VideoRecognizer(PhotoOCR(cfg, RegionFinder(d, detector, d.processing_mode), engine), bib_filter)

    vstate = {"first": None, "last": 0.0}

    def progress(i: int, n: int, path: str) -> None:
        stop_if_cancelled()
        now = time.perf_counter()
        if vstate["first"] is None:
            vstate["first"] = now
        done = i - 1
        if i == 1 or i == n or now - vstate["last"] >= 1 / MAX_MESSAGES_PER_SECOND:
            vstate["last"] = now
            per = (now - vstate["first"]) / done if done else None
            send(proto.VIDEO_ANALYSING, current=i, completed=done, total=n, filename=os.path.basename(path),
                 speed=round(per, 3) if per else None, remaining=round(per * (n - done), 1) if per else None)

    return analyze_videos(conn, folder, event_id, cfg, get_recognizer, photo_settings, years,
                          progress=progress, thumbnail_cache=cfg.paths.thumbnail_cache)


def simulate(folder: str, event: str, delay: float, report_modules: bool = False,
             watcher: CancelWatcher | None = None) -> int:
    """Tests only (Phase 7.3A dry run): scan, count, a delay per photo; nothing is opened or written."""
    watcher = watcher or CancelWatcher()
    start = time.perf_counter()
    if not os.path.isdir(folder):
        send(proto.ERROR, message=f"Folder not found: {folder}")
        return 1
    from config import get_config
    from scanner.photo_scanner import iter_photo_files
    cfg = get_config()
    send(proto.SCANNING, total=0, found=0, folder=folder, event=event)
    paths = []
    for path, _st in iter_photo_files(folder, cfg.scan.extensions, cfg.scan.recursive, cfg.scan.skip_dir_names,
                                      (cfg.paths.default_export_root, cfg.paths.data_dir)):
        paths.append(path)
        if watcher.cancelled.is_set():
            break
    total, scan_cut = len(paths), watcher.cancelled.is_set()
    send(proto.SCANNING, total=total, found=total, folder=folder, event=event)
    done, run_start, last = 0, time.perf_counter(), 0.0
    for path in paths:
        if watcher.cancelled.is_set():
            break
        time.sleep(delay)
        done += 1
        now = time.perf_counter()
        if done == total or now - last >= 1 / MAX_MESSAGES_PER_SECOND:
            last = now
            per_photo = (now - run_start) / done
            send(proto.ANALYSING, current=done, completed=done, total=total, filename=os.path.basename(path),
                 speed=round(per_photo, 4), remaining=round(per_photo * (total - done), 1))
    extra = {"modules": sorted({m.split(".")[0] for m in sys.modules})} if report_modules else {}
    status = proto.FINISHED if done == total and not scan_cut else proto.CANCELLED
    send(status, total=total, processed=done, failed=0, duration=round(time.perf_counter() - start, 2),
         simulated=True, **extra)
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    _protect_protocol_stream()
    p = argparse.ArgumentParser(prog="analysis_worker", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("gpu-check")
    a = sub.add_parser("analyse")
    a.add_argument("--folder", required=True)
    a.add_argument("--event", required=True)
    a.add_argument("--db", help="index database (default: config, data\\runner_photo_finder.db)")
    a.add_argument("--thumbnail-cache", help="thumbnail folder (default: config, data\\thumbnails)")
    a.add_argument("--limit", type=int, help="analyse at most N photos in this run (as index_cli --limit)")
    a.add_argument("--prefetch", type=int, default=None,
                   help=f"photos decoded ahead (default {DEFAULT_PREFETCH}; 0 = off; env RUNNERFINDER_PREFETCH)")
    a.add_argument("--pipeline", choices=PIPELINES, default=None,
                   help=f"analysis pipeline (default {DEFAULT_PIPELINE}; env RUNNERFINDER_PIPELINE)")
    a.add_argument("--no-videos", action="store_true", help="V1.3: analyse photos only (skip the video stage)")
    a.add_argument("--simulate", type=float, metavar="SECONDS", help=argparse.SUPPRESS)   # tests only
    a.add_argument("--report-modules", action="store_true", help="list loaded modules in the final message")
    args = p.parse_args(argv)
    try:
        if args.command == "gpu-check":
            return gpu_check()
        if args.simulate is not None:
            return simulate(args.folder, args.event, args.simulate, args.report_modules)
        prefetch = args.prefetch
        if prefetch is None:
            prefetch = int(os.environ.get("RUNNERFINDER_PREFETCH", DEFAULT_PREFETCH))
        pipeline = args.pipeline or os.environ.get("RUNNERFINDER_PIPELINE", DEFAULT_PIPELINE)
        if pipeline not in PIPELINES:
            pipeline = DEFAULT_PIPELINE
        if pipeline == "s3" and prefetch > 0:
            # Idle OpenMP threads (torch and Paddle share the Intel OpenMP runtime) sleep at once instead of
            # spinning for 200 ms, leaving the cores to the three stages. Scheduling only: V1.3 Phase B showed
            # byte-identical results with and without it. Must be set before torch / Paddle are imported
            # (they are imported in analyse()). An explicit KMP_BLOCKTIME in the environment wins.
            os.environ.setdefault("KMP_BLOCKTIME", "0")
        return analyse(args.folder, args.event, args.db, args.thumbnail_cache, args.limit, args.report_modules,
                       prefetch=max(0, prefetch), pipeline=pipeline, videos=not args.no_videos)
    except Exception as exc:  # noqa: BLE001 - always end with a message the GUI understands
        import traceback
        traceback.print_exc()                     # to stderr (kept by AnalysisJob)
        send(proto.ERROR, message=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
