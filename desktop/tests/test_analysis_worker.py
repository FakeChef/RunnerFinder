"""Phase 7.3A worker tests: protocol, worker process, AnalysisJob, cancel, responsiveness.
(7.3B: the dry run is the worker's test-only --simulate option.)

The worker runs in its own process with the GPU environment's Python (as in the app).
Dry run only: no OCR, nothing written.
Run from the repository folder:
    <python with PySide6> -m unittest desktop.tests.test_analysis_worker -v
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop.services import progress_protocol as proto  # noqa: E402
from desktop.services.analysis_job import GPU_VENV_PYTHON, AnalysisJob, worker_python  # noqa: E402
from desktop.tests.private_data import TEST_PHOTOS, needs  # noqa: E402

PHOTOS = TEST_PHOTOS                                    # 370 photos, read-only
FORBIDDEN_IN_WORKER = {"paddle", "paddleocr", "paddlex", "torch", "ultralytics", "cv2",
                       "ocr", "detection", "indexing", "database", "PySide6"}
APP = QApplication.instance() or QApplication(sys.argv)


def pump_until(condition, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        APP.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class JobRecorder:
    def __init__(self, job):
        self.messages, self.exit = [], None
        job.message.connect(self.messages.append)
        job.finished.connect(lambda code, err: setattr(self, "exit", (code, err)))

    def of(self, status):
        return [m for m in self.messages if m["status"] == status]


class ProtocolTest(unittest.TestCase):
    def test_round_trip_and_noise_lines(self):
        line = proto.encode(proto.ANALYSING, current=123, total=6974, filename="IMG_0123.JPG", speed=0.8)
        self.assertEqual(proto.decode(line), {"status": "analysing", "current": 123, "total": 6974,
                                              "filename": "IMG_0123.JPG", "speed": 0.8})
        self.assertEqual(proto.decode(line.encode() + b"\r\n")["current"], 123)
        for noise in ("", "W1014 paddle warning", "{not json", '{"no": "status"}', "[1, 2]"):
            self.assertIsNone(proto.decode(noise))


class WorkerTest(unittest.TestCase):
    def run_job(self, args, timeout=120):
        job = AnalysisJob()
        rec = JobRecorder(job)
        job.start(args)
        self.assertTrue(pump_until(lambda: rec.exit is not None, timeout), rec.messages[-3:])
        return job, rec

    def test_worker_uses_the_gpu_environment(self):
        gpu_python = GPU_VENV_PYTHON
        if gpu_python.is_file() and not os.environ.get("RUNNERFINDER_WORKER_PYTHON"):
            self.assertEqual(Path(worker_python()), gpu_python)

    @needs(PHOTOS, "RUNNERFINDER_TEST_PHOTOS")
    def test_worker_starts_scans_and_sends_progress(self):
        _job, rec = self.run_job(["analyse", "--folder", str(PHOTOS), "--event", "t", "--simulate", "0.001",
                                  "--report-modules"])
        self.assertEqual(rec.exit[0], 0, rec.exit[1])
        scans = rec.of(proto.SCANNING)
        self.assertEqual(scans[0]["total"], 0)
        self.assertEqual(scans[-1]["total"], 370)                    # = 370-photo validation set
        steps = rec.of(proto.ANALYSING)
        self.assertGreater(len(steps), 1)
        self.assertEqual([s["current"] for s in steps], sorted(s["current"] for s in steps))
        self.assertEqual(steps[-1]["current"], 370)
        self.assertTrue(all(s["filename"] and s["speed"] > 0 for s in steps))
        final = rec.messages[-1]
        self.assertEqual((final["status"], final["processed"], final["total"]), ("finished", 370, 370))
        self.assertFalse(set(final["modules"]) & FORBIDDEN_IN_WORKER, set(final["modules"]) & FORBIDDEN_IN_WORKER)

    @needs(PHOTOS, "RUNNERFINDER_TEST_PHOTOS")
    def test_cancel_stops_the_worker(self):
        job = AnalysisJob()
        rec = JobRecorder(job)
        job.start(["analyse", "--folder", str(PHOTOS), "--event", "t", "--simulate", "0.05"])
        self.assertTrue(pump_until(lambda: rec.of(proto.ANALYSING), 60))
        t0 = time.monotonic()
        job.cancel()
        self.assertTrue(pump_until(lambda: rec.exit is not None, 10))
        self.assertLess(time.monotonic() - t0, 5)
        self.assertFalse(job.running)
        final = rec.messages[-1]
        self.assertEqual(final["status"], "cancelled")
        self.assertLess(final["processed"], 370)

    @needs(PHOTOS, "RUNNERFINDER_TEST_PHOTOS")
    def test_worker_stops_when_the_gui_goes_away(self):
        """stdin closed (GUI crashed or closed) = cancel: no orphan worker."""
        proc = subprocess.Popen([worker_python(), "-m", "desktop.services.analysis_worker", "analyse",
                                 "--folder", str(PHOTOS), "--event", "t", "--simulate", "0.05"],
                                cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        first = proto.decode(proc.stdout.readline())
        self.assertEqual(first["status"], "scanning")
        proc.stdin.close()
        out, _ = proc.communicate(timeout=30)
        last = [proto.decode(x) for x in out.splitlines() if proto.decode(x)][-1]
        self.assertEqual(last["status"], "cancelled")

    def test_missing_folder_is_an_error_message(self):
        missing = str(Path(tempfile.gettempdir()) / "runnerfinder no such folder")
        _job, rec = self.run_job(["analyse", "--folder", missing, "--event", "t"])
        self.assertEqual(rec.messages[-1]["status"], "error")
        self.assertIn("Folder not found", rec.messages[-1]["message"])

    @needs(PHOTOS, "RUNNERFINDER_TEST_PHOTOS")
    def test_gui_stays_responsive_during_a_run(self):
        job = AnalysisJob()
        rec = JobRecorder(job)
        ticks = []
        timer = QTimer()
        timer.setInterval(20)
        timer.timeout.connect(lambda: ticks.append(time.monotonic()))
        timer.start()
        job.start(["analyse", "--folder", str(PHOTOS), "--event", "t", "--simulate", "0.005"])
        self.assertTrue(pump_until(lambda: rec.exit is not None, 120))
        timer.stop()
        gaps = [b - a for a, b in zip(ticks, ticks[1:])]
        self.assertGreater(len(ticks), 20)
        self.assertLess(max(gaps), 0.5, f"event loop blocked for {max(gaps):.2f} s")
        self.assertLessEqual(len(rec.of(proto.ANALYSING)), 370)      # throttled, never one line per photo flood

    def test_gpu_check_is_environment_only(self):
        _job, rec = self.run_job(["gpu-check"])
        gpu = rec.of(proto.GPU)
        self.assertEqual(len(gpu), 1, rec.exit)
        if gpu[0]["available"]:
            self.assertTrue(gpu[0]["name"] and gpu[0]["cuda"] and gpu[0]["paddle"])
        # PaddleOCR / YOLO are never imported by the GPU check (checked in a fresh worker process)
        code = ("import sys, json; from desktop.services import analysis_worker as w; w.gpu_check(); "
                "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))")
        out = subprocess.run([worker_python(), "-c", code], cwd=str(ROOT), capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120)
        mods = set(json.loads(out.stdout.strip().splitlines()[-1]))
        self.assertFalse(mods & {"paddleocr", "paddlex", "ultralytics", "torch", "ocr", "detection"},
                         mods & {"paddleocr", "paddlex", "ultralytics", "torch", "ocr", "detection"})


if __name__ == "__main__":
    unittest.main()
