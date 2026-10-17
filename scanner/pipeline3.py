"""
Three-stage photo pipeline for the analysis loop (V1.3, "S3").

    stage A  PhotoPrefetcher (V1.2, unchanged): read + JPEG decode + EXIF + RGB, thumbnail hook
    stage B  detection thread: RegionFinder.find() (resize, YOLO26n on the CPU, ranking, regions)
    stage C  the analysis loop (caller's thread): OCR on the GPU, filter, ordered SQLite commit

Stages A and B work on the next photos while stage C reads the current one. The SAME functions
run on the same photos in the same order; only the thread and the time at which a photo is
decoded and detected change, so stored results do not change.

    with DetectionPipeline(paths, finder.find, on_decoded, depth=2) as photos:
        for path in paths:
            image, orientation, decode_ms, detection = photos.take(path)   # re-raises its error

* Identical YOLO numerics: in V1.2 YOLO runs in the analysis thread, where PaddleOCR has set the
  OpenMP thread count to 1 (it does so at load and at every OCR call; torch and Paddle share the
  Intel OpenMP runtime). Stage B sets exactly that - omp_set_num_threads(1) on the shared runtime -
  before every detection. torch.set_num_threads(1) is NOT equivalent (V1.3 Phase A: 231 of 370
  photos differ in the 7th decimal), so it is not used.
* Order: one producer per stage, FIFO queues; every hand-over checks the expected path.
* Errors: a decode error (stage A) or a detection error (stage B) travels with its photo and is
  raised by take() for that photo only, so the analysis loop's existing except-blocks handle it
  exactly as before (V1.2: both raised inside the loop's try).
* Memory: at most depth photos queued per stage plus one being worked on by A, B and C each
  (2 + 1 + 2 + 1 + 1 = 7 decoded photos for depth 2).
* Stop: close() (also on cancel / error in the loop) stops B, closes A, drops every queued photo
  and joins both threads. Nothing is committed here: only the analysis loop commits, so photos
  that were only decoded or detected are never stored.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable

from scanner.prefetch import PhotoPrefetcher, _STOP_POLL_S
from scanner.photo_scanner import load_image_with_orientation


def paddle_equivalent_single_thread() -> Callable[[], None] | None:
    """omp_set_num_threads(1) on the Intel OpenMP runtime loaded in this process (libiomp5md.dll, shared
    by torch and Paddle), i.e. what PaddleOCR does to the analysis thread in V1.2. None if that runtime
    is not loaded (e.g. tests with stand-in detectors)."""
    hook, _reason = openmp_single_thread_hook()
    return hook


#: the OpenMP setting validated for S3 in V1.3 Phase B (scheduling only, byte-identical results)
VALIDATED_KMP_BLOCKTIME = "0"


def openmp_single_thread_hook() -> tuple[Callable[[], None] | None, str]:
    """(hook, reason). The hook is omp_set_num_threads(1) on the Intel OpenMP runtime that torch/Paddle
    ALREADY loaded (GetModuleHandle: never a second copy found on PATH). hook None: reason says why."""
    import ctypes
    import sys
    if "torch" not in sys.modules:
        return None, "torch not loaded"
    if sys.platform != "win32":
        return None, "not Windows"
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        handle = kernel32.GetModuleHandleW("libiomp5md.dll")
        if not handle:
            return None, "libiomp5md.dll is not loaded in this process"
        buf = ctypes.create_unicode_buffer(1024)
        kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
        kernel32.GetModuleFileNameW(ctypes.c_void_p(handle), buf, 1024)
        lib = ctypes.CDLL(buf.value)                  # the same, already loaded module
        fn = lib.omp_set_num_threads
    except (OSError, AttributeError) as exc:
        return None, f"OpenMP runtime unusable: {exc}"
    return (lambda: fn(1)), f"libiomp5md.dll at {buf.value}"


def s3_preconditions() -> tuple[bool, str]:
    """True only with the configuration validated in V1.3 Phase B: the shared Intel OpenMP runtime is
    loaded and usable, and KMP_BLOCKTIME is the validated value (set by the worker before torch/Paddle
    load). Anything else is unverified: the caller must fall back to the V1.2 prefetch pipeline."""
    import os
    hook, reason = openmp_single_thread_hook()
    if hook is None:
        return False, reason
    kb = os.environ.get("KMP_BLOCKTIME")
    if kb != VALIDATED_KMP_BLOCKTIME:
        return False, f"KMP_BLOCKTIME={kb!r} (validated: {VALIDATED_KMP_BLOCKTIME!r})"
    return True, reason


class DetectionPipeline:
    def __init__(self, paths: list[str], detect: Callable[[object], object],
                 on_decoded: Callable[[str, object], None] | None = None, depth: int = 2,
                 loader: Callable = load_image_with_orientation,
                 before_detect: Callable[[], None] | None = None):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self._paths = list(paths)
        self._detect = detect
        self._before_detect = before_detect
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self.detected = 0                      # photos detected by stage B (tests, statistics)
        self._decoded = PhotoPrefetcher(self._paths, on_decoded, depth=depth, loader=loader)   # stage A
        self._thread = threading.Thread(target=self._run, name="photo-detect", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ stage B
    def _run(self) -> None:
        for path in self._paths:
            if self._stop.is_set():
                return
            image = orientation = decode_ms = detection = exc = None
            try:
                image, orientation, decode_ms = self._decoded.take(path)
            except BaseException as e:  # noqa: BLE001 - the decode error belongs to this photo
                exc = e
            if exc is None:
                try:
                    if self._before_detect is not None:
                        self._before_detect()
                    detection = self._detect(image)
                except BaseException as e:  # noqa: BLE001 - handed to the analysis loop unchanged
                    exc = e
            self.detected += 1
            item = (path, image, orientation, decode_ms, detection, exc)
            while not self._stop.is_set():
                try:
                    self._queue.put(item, timeout=_STOP_POLL_S)
                    break
                except queue.Full:
                    continue
            if self._stop.is_set():
                return

    # ------------------------------------------------------------------ stage C (caller)
    def take(self, path: str):
        """(image, orientation, decode_ms, detection) of `path`, the next photo in order; re-raises its error."""
        while True:
            try:
                got, image, orientation, decode_ms, detection, exc = self._queue.get(timeout=_STOP_POLL_S)
                break
            except queue.Empty:
                if not self._thread.is_alive() and self._queue.empty():
                    raise RuntimeError(f"photo pipeline stopped before {path}")
        if got != path:
            raise RuntimeError(f"photo pipeline out of order: expected {path}, got {got}")
        if exc is not None:
            raise exc
        return image, orientation, decode_ms, detection

    def close(self) -> None:
        """Stop both stages and drop queued photos (they were never OCR'd or committed)."""
        self._stop.set()
        self._decoded.close()                  # stage A stops; a waiting take() in B ends
        self._drain()
        self._thread.join(timeout=60)
        self._drain()                          # anything put between the drain and the join

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def alive(self) -> bool:
        return self._thread.is_alive() or self._decoded._thread.is_alive()

    def __enter__(self) -> "DetectionPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

