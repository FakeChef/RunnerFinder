r"""
Environment check for Runner Photo Finder.

    .\.venv\Scripts\python.exe check_environment.py           full install
    .\.venv\Scripts\python.exe check_environment.py --core    scanner/database/tests only
    .\.venv\Scripts\python.exe check_environment.py --models  full install, plus download
                                                              the AI models (once) and run
                                                              YOLO + PaddleOCR on a test image

Checks: Python version, installed versions against the lock file,
`pip check`, imports, the shared OpenCV module, and GPU availability.
Exit code 0 = no failures.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import config  # sets YOLO_AUTOINSTALL / PaddleOCR flags before any AI import

ROOT = Path(__file__).resolve().parent
FULL_LOCK = ROOT / "requirements-lock-win-py311.txt"
CORE_LOCK = ROOT / "requirements-core-lock-win-py311.txt"
OPENCV_REPAIR = ROOT / "requirements-opencv-repair-win-py311.txt"
PY = r".\.venv\Scripts\python.exe"

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
failures = 0


def report(status: str, message: str) -> None:
    global failures
    failures += status == FAIL
    print(f"{status} {message}")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# ------------------------------------------------------------------ checks

def check_python() -> None:
    v, bits = sys.version_info, platform.architecture()[0]
    if v < (3, 11) or bits != "64bit":
        report(FAIL, f"Python {v.major}.{v.minor} {bits}: need 64-bit Python 3.11")
    elif v[:2] != (3, 11):
        report(WARN, f"Python {v.major}.{v.minor}: the lock files are made for 3.11")
    else:
        report(OK, f"Python {platform.python_version()} ({bits})")
    if platform.system() != "Windows":
        report(WARN, f"Running on {platform.system()}: the lock files target Windows")


def check_lock(lock: Path) -> None:
    pins = dict((norm(n), v) for n, v in
                re.findall(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)", lock.read_text(), re.M))
    bad = []
    for name, wanted in sorted(pins.items()):
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            have = None
        if have != wanted:
            bad.append(f"{name}: locked {wanted}, installed {have or 'MISSING'}")
    if bad:
        report(FAIL, f"{len(bad)} of {len(pins)} packages differ from {lock.name}:")
        for line in bad[:15]:
            print(f"         {line}")
        if len(bad) > 15:
            print(f"         ... and {len(bad) - 15} more")
    else:
        report(OK, f"All {len(pins)} packages match {lock.name}")


def check_pip() -> None:
    r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if r.returncode == 0:
        report(OK, "pip check: no broken requirements")
    else:
        report(FAIL, "pip check found problems:\n         "
               + (r.stdout + r.stderr).strip().replace("\n", "\n         "))


def check_import(label: str, statement: str) -> bool:
    try:
        exec(statement, {})
        report(OK, f"import {label}")
        return True
    except Exception as exc:  # noqa: BLE001 - report every kind of failure
        report(FAIL, f"import {label}: {type(exc).__name__}: {exc}")
        return False


def check_core_imports() -> None:
    import sqlite3
    check_import("PIL (Pillow)", "from PIL import Image, ImageOps")
    check_import("pytest", "import pytest")
    report(OK, f"SQLite {sqlite3.sqlite_version}")
    check_import("project modules", "import scanner.folder_sync, database.database, ocr.bib_format")


def check_pillow_open() -> None:
    """Run after the AI imports: Ultralytics replaces PIL.Image.open."""
    from utils import pil_compat
    msg = pil_compat.status()
    report(WARN if "before it could be saved" in msg else OK, msg)


def check_ai_imports() -> None:
    check_import("streamlit", "import streamlit")
    check_import("torch", "import torch")
    check_import("ultralytics.YOLO", "from ultralytics import YOLO")
    check_import("paddle", "import paddle")
    if check_import("paddleocr.PaddleOCR", "from paddleocr import PaddleOCR"):
        from paddlex.utils.deps import is_extra_available
        if is_extra_available("ocr-core"):
            report(OK, "PaddleX accepts the installed OCR dependencies")
        else:
            report(FAIL, "PaddleX reports missing OCR dependencies (often OpenCV)")


def check_ultralytics_privacy() -> None:
    try:
        from detection.person_detector import disable_ultralytics_analytics
        from ultralytics import settings
        disable_ultralytics_analytics()
        if settings.get("sync"):
            report(FAIL, "Ultralytics usage analytics could not be switched off")
        else:
            offline = "offline mode on" if os.environ.get("YOLO_OFFLINE") else \
                "offline mode starts once models\\yolo26n.pt exists"
            report(OK, f"Ultralytics analytics off ({offline})")
    except Exception as exc:  # noqa: BLE001
        report(FAIL, f"Ultralytics privacy check: {type(exc).__name__}: {exc}")


def check_opencv() -> None:
    """Both OpenCV packages share one cv2 folder; make sure it is intact."""
    versions = {}
    for dist in ("opencv-python", "opencv-contrib-python"):
        try:
            versions[dist] = md.version(dist)
        except md.PackageNotFoundError:
            versions[dist] = None
    repair = (f"Repair with:\n         {PY} -m pip install --force-reinstall --no-deps "
              f"--require-hashes -r {OPENCV_REPAIR.name}")
    if None in versions.values() or len(set(versions.values())) != 1:
        report(FAIL, f"OpenCV packages inconsistent {versions}. {repair}")
        return
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        report(FAIL, f"cv2 cannot be imported ({exc}). {repair}")
        return
    expected = versions["opencv-python"].rsplit(".", 1)[0]       # 4.10.0.84 -> 4.10.0
    if cv2.__version__ != expected:
        report(FAIL, f"cv2 is {cv2.__version__}, expected {expected}. {repair}")
    else:
        report(OK, f"OpenCV {cv2.__version__} (both packages at {versions['opencv-python']})")


def check_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            report(OK, f"YOLO can use GPU: {torch.cuda.get_device_name(0)}")
        else:
            report(OK, "YOLO will run on CPU (no CUDA GPU visible to PyTorch)")
    except Exception:  # noqa: BLE001
        pass
    try:
        import paddle
        gpu = paddle.device.is_compiled_with_cuda()
        report(OK, "PaddleOCR GPU build" if gpu else "PaddleOCR will run on CPU")
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ model test

def make_test_image():
    """A white 'bib' with a black number on grey."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (640, 480), (90, 90, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([120, 140, 520, 340], fill=(255, 255, 255))
    try:
        font = ImageFont.truetype("arialbd.ttf", 110)       # Windows Arial Bold
    except OSError:
        font = ImageFont.load_default(size=110)
    draw.text((150, 175), "T1234", fill=(0, 0, 0), font=font)
    return np.array(img)


def model_test() -> None:
    """
    Smoke test only: proves the models load and run. One synthetic image
    says nothing about accuracy or speed on race photos.
    """
    cfg = config.get_config()
    image = make_test_image()
    print("\n--- YOLO (first run downloads the weights) ---")
    try:
        from ultralytics import YOLO
        weights = cfg.paths.models_dir / cfg.detection.model_name
        model = YOLO(str(weights))
        model.predict(image, verbose=False)               # warm-up
        t0 = time.perf_counter()
        model.predict(image, verbose=False)
        report(OK, f"YOLO ran ({weights.name}); one 640x480 image: "
                   f"{time.perf_counter() - t0:.3f} s after warm-up")
    except Exception as exc:  # noqa: BLE001
        report(FAIL, f"YOLO test: {type(exc).__name__}: {exc}")

    print("\n--- PaddleOCR (first run downloads the models) ---")
    print(f"       model cache: {os.environ.get('PADDLE_PDX_CACHE_HOME')}")
    try:
        # The same reader class and settings as the application (config.OCRConfig).
        from ocr.bib_reader import PaddleOCREngine
        engine = PaddleOCREngine(cfg.ocr)
        from PIL import Image
        pil = Image.fromarray(image)
        engine.read(pil)                                   # warm-up
        t0 = time.perf_counter()
        lines = engine.read(pil)
        elapsed = time.perf_counter() - t0
        texts = [ln.text for ln in lines]
        if any("T1234" in t.replace(" ", "").upper() for t in texts):
            report(OK, f"PaddleOCR read {texts}; one image: {elapsed:.3f} s after warm-up "
                       f"({engine.name})")
        else:
            report(FAIL, f"PaddleOCR ran but read {texts} instead of 'T1234'")
    except Exception as exc:  # noqa: BLE001
        report(FAIL, f"PaddleOCR test: {type(exc).__name__}: {exc}")
        print("       If the model download is blocked, choose another source first:\n"
              "         Command Prompt:  set PADDLE_PDX_MODEL_SOURCE=bos\n"
              '         PowerShell:      $env:PADDLE_PDX_MODEL_SOURCE = "bos"')


def check_paddle_cache() -> None:
    cache = os.environ.get("PADDLE_PDX_CACHE_HOME", "")
    if not cache:
        report(FAIL, "PADDLE_PDX_CACHE_HOME is not set")
    elif not os.path.isabs(cache):
        report(FAIL, f"PADDLE_PDX_CACHE_HOME is not absolute: {cache}")
    elif Path(cache).resolve() != config.PADDLE_CACHE_DIR.resolve():
        report(WARN, f"PADDLE_PDX_CACHE_HOME overridden by environment: {cache}")
    else:
        report(OK, f"PaddleOCR model cache: {cache}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--core", action="store_true", help="check the core install only")
    group.add_argument("--models", action="store_true", help="also download and run the models")
    args = parser.parse_args()

    print("Runner Photo Finder - environment check")
    print(f"System: {platform.platform()} | CPU: {platform.processor() or platform.machine()} "
          f"| logical cores: {os.cpu_count()}\n")
    check_python()
    check_lock(CORE_LOCK if args.core else FULL_LOCK)
    check_pip()
    check_core_imports()
    if not args.core:
        check_ai_imports()
        check_pillow_open()
        check_ultralytics_privacy()
        check_paddle_cache()
        check_opencv()
        check_gpu()
    if args.models:
        model_test()
    print("\nAll checks passed." if failures == 0 else f"\n{failures} check(s) FAILED.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
