"""
Photo prefetch for the analysis loop (V1.2).

A producer thread decodes the next photos while the current photo is in
person detection and OCR. It runs the SAME functions the analysis loop ran
inline in V1.1 - load_image_with_orientation() (read, JPEG decode, EXIF
orientation, RGB) and the optional on_decoded() hook (thumbnail) - on the
same photos in the same order, so the analysis results do not change; only
the time at which a photo is decoded does.

    with PhotoPrefetcher(paths, on_decoded, depth=2) as photos:
        for path in paths:
            image, orientation, decode_ms = photos.take(path)   # re-raises a decode error

* Order: one producer, one FIFO queue; take() checks that the queued photo
  is the one the loop expects.
* Memory: at most `depth` decoded photos wait in the queue, plus the one the
  producer is finishing (depth + 1 ahead of the photo being analysed).
* Stop: close() (also on cancel / Ctrl+C / any error in the loop) stops the
  producer, drops the queued photos and waits for the thread. Nothing is
  committed here: committing stays in the analysis loop, so photos that were
  only prefetched are never stored.
* Pillow releases the GIL while decoding, rotating and converting, so a
  thread is enough (no extra process, no copies of the image).
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable

from scanner.photo_scanner import load_image_with_orientation

_STOP_POLL_S = 0.1


class PhotoPrefetcher:
    def __init__(self, paths: list[str], on_decoded: Callable[[str, object], None] | None = None,
                 depth: int = 2, loader: Callable = load_image_with_orientation):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self._paths = list(paths)
        self._on_decoded = on_decoded
        self._loader = loader
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self.produced = 0                      # photos decoded by the producer (tests, statistics)
        self._thread = threading.Thread(target=self._produce, name="photo-prefetch", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ producer
    def _produce(self) -> None:
        for path in self._paths:
            if self._stop.is_set():
                return
            t = time.perf_counter()
            try:
                image, orientation = self._loader(path)
                item = (path, image, orientation, (time.perf_counter() - t) * 1000, None)
            except BaseException as exc:  # noqa: BLE001 - handed to the analysis loop unchanged
                item = (path, None, None, (time.perf_counter() - t) * 1000, exc)
            if item[4] is None and self._on_decoded is not None:
                try:
                    self._on_decoded(path, item[1])
                except Exception:  # noqa: BLE001 - as in V1.1: a thumbnail problem must not fail the photo
                    pass
            self.produced += 1
            while not self._stop.is_set():
                try:
                    self._queue.put(item, timeout=_STOP_POLL_S)
                    break
                except queue.Full:
                    continue

    # ------------------------------------------------------------------ consumer
    def take(self, path: str):
        """(image, orientation, decode_ms) of `path`, the next photo in order; re-raises its decode error."""
        while True:
            try:
                got, image, orientation, decode_ms, exc = self._queue.get(timeout=_STOP_POLL_S)
                break
            except queue.Empty:
                if not self._thread.is_alive() and self._queue.empty():
                    raise RuntimeError(f"photo prefetch stopped before {path}")
        if got != path:
            raise RuntimeError(f"photo prefetch out of order: expected {path}, got {got}")
        if exc is not None:
            raise exc
        return image, orientation, decode_ms

    def close(self) -> None:
        """Stop the producer and drop queued photos (they were never analysed or committed)."""
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=30)
        while True:                            # anything put between the drain and the join
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def __enter__(self) -> "PhotoPrefetcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
