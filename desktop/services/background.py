"""
Background work for the services without per-task Qt objects.

fn runs on a plain Python thread (concurrent.futures); its result comes back
to the GUI thread through ONE long-lived QObject signal (queued connection).
This replaces per-task QRunnable + QObject pairs, whose C++ objects Qt can
delete while Python still holds them (access violation seen in Phase 7.3A tests).
"""
from __future__ import annotations

import concurrent.futures as cf

from PySide6.QtCore import QObject, Signal, Slot


class BackgroundRunner(QObject):
    _delivered = Signal(object, object)          # callback, value (always handled in the GUI thread)

    def __init__(self, parent: QObject | None = None, threads: int = 1):
        super().__init__(parent)
        self._executor = cf.ThreadPoolExecutor(max_workers=threads, thread_name_prefix="rf-background")
        self._futures: set[cf.Future] = set()
        self._delivered.connect(self._deliver)

    def run(self, fn, on_done, on_failed=None) -> None:
        """fn() in the background; then on_done(result) or on_failed(message) in the GUI thread."""
        def job():
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 - shown to the user, never crashes the window
                if on_failed is not None:
                    self._delivered.emit(on_failed, f"{type(exc).__name__}: {exc}")
                return
            self._delivered.emit(on_done, result)
        future = self._executor.submit(job)
        self._futures.add(future)
        future.add_done_callback(self._futures.discard)

    def wait_idle(self, msecs: int = 30000) -> bool:
        """Wait until the submitted work has run (results are delivered by the event loop)."""
        _done, pending = cf.wait(list(self._futures), timeout=msecs / 1000)
        return not pending

    @Slot(object, object)
    def _deliver(self, callback, value) -> None:      # bound slot: runs in this object's (GUI) thread
        callback(value)
