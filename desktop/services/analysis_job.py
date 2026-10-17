"""
AnalysisJob: one worker process seen from the GUI (Phase 7.3A).

The GUI process only starts the worker (QProcess), reads its JSON lines and
writes "cancel". Heavy imports (PaddlePaddle, later YOLO / PaddleOCR) happen in
the worker process, never here; if the worker crashes the window stays.

    job = AnalysisJob(parent)
    job.message.connect(...)          # dict, see services/progress_protocol.py
    job.finished.connect(...)         # exit code, last stderr lines
    job.start(["analyse", "--folder", folder, "--event", event])
    job.cancel()
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from desktop.services import progress_protocol as proto

PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: optional GPU environment inside the checkout (the installed app sets RUNNERFINDER_WORKER_PYTHON instead)
GPU_VENV_PYTHON = PROJECT_ROOT / ".venv-gpu" / "Scripts" / "python.exe"
STDERR_TAIL_CHARS = 4000


def worker_python() -> str:
    """Python for the worker: RUNNERFINDER_WORKER_PYTHON, else .venv-gpu, else this Python."""
    env = os.environ.get("RUNNERFINDER_WORKER_PYTHON")
    if env:
        return env
    if GPU_VENV_PYTHON.is_file():
        return str(GPU_VENV_PYTHON)
    exe = Path(sys.executable)
    console = exe.with_name("python.exe")            # pythonw.exe has no stdout
    return str(console if exe.name.lower() == "pythonw.exe" and console.is_file() else exe)


class AnalysisJob(QObject):
    started = Signal()
    message = Signal(dict)                  # one protocol message
    finished = Signal(int, str)             # exit code (-1: crashed / killed), stderr tail

    def __init__(self, parent: QObject | None = None, python: str | None = None,
                 cancel_grace_ms: int = 30000):
        super().__init__(parent)
        self.python = python or worker_python()
        self.cancel_grace_ms = cancel_grace_ms
        self._buffer = b""
        self._stderr = ""
        self.final: dict | None = None      # last finished / cancelled / error message
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(PROJECT_ROOT))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self._proc.setProcessEnvironment(env)
        # No console window: Qt starts the worker with CREATE_NO_WINDOW when the GUI
        # itself has no console (pythonw.exe / RunnerFinder.pyw).
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.started.connect(self.started)
        self._proc.finished.connect(self._finished)
        self._proc.errorOccurred.connect(self._error)
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self.kill)
        self._done = False

    @property
    def running(self) -> bool:
        return self._proc.state() != QProcess.ProcessState.NotRunning

    @property
    def pid(self) -> int:
        return int(self._proc.processId())

    def start(self, args: list[str]) -> None:
        self._buffer, self._stderr, self.final, self._done = b"", "", None, False
        self._proc.start(self.python, ["-u", "-m", "desktop.services.analysis_worker", *args])

    def cancel(self) -> None:
        """Ask the worker to stop after the current photo; kill it if it does not within the grace time."""
        if self.running:
            self._proc.write((proto.CANCEL_COMMAND + "\n").encode())
            self._kill_timer.start(self.cancel_grace_ms)

    def kill(self) -> None:
        if self.running:
            self._proc.kill()

    def wait(self, msecs: int = 30000) -> bool:
        """For tests and shutdown."""
        return self._proc.waitForFinished(msecs) or not self.running

    # ------------------------------------------------------------------ process output
    def _read_stdout(self) -> None:
        self._buffer += bytes(self._proc.readAllStandardOutput())
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            msg = proto.decode(line)
            if msg is not None:
                if msg["status"] in proto.FINAL:
                    self.final = msg
                self.message.emit(msg)

    def _read_stderr(self) -> None:
        text = bytes(self._proc.readAllStandardError()).decode("utf-8", errors="replace")
        self._stderr = (self._stderr + text)[-STDERR_TAIL_CHARS:]

    def _finished(self, code: int, status) -> None:
        self._kill_timer.stop()
        self._read_stdout()
        if self._buffer:
            self._buffer += b"\n"
            self._read_stdout()
        self._read_stderr()
        crashed = status == QProcess.ExitStatus.CrashExit
        self._emit_finished(-1 if crashed else code)

    def _error(self, err) -> None:
        if err == QProcess.ProcessError.FailedToStart:
            self._stderr += f"\nCannot start the worker: {self.python} ({self._proc.errorString()})"
            self._emit_finished(-1)

    def _emit_finished(self, code: int) -> None:
        if not self._done:
            self._done = True
            self.finished.emit(code, self._stderr.strip())
