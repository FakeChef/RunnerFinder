"""
Real PaddleOCR (needs the full install). The settings tests run without the
OCR models; the reading test runs only when the models are in the project
cache (models/paddlex/official_models), i.e. after
`check_environment.py --models` has downloaded them.
"""
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("paddleocr")

import config  # noqa: E402
from config import OCRConfig  # noqa: E402
from ocr.bib_reader import (PaddleOCREngine, paddleocr_init_kwargs,  # noqa: E402
                            paddleocr_predict_kwargs)

ROOT = Path(__file__).resolve().parents[2]
MODELS = config.PADDLE_CACHE_DIR / "official_models"
HAVE_MODELS = all((MODELS / m).is_dir() for m in ("PP-OCRv5_mobile_det", "en_PP-OCRv5_mobile_rec"))


def test_paddleocr_accepts_our_settings():
    from paddleocr import PaddleOCR
    from paddleocr._common_args import parse_common_args   # validates device/mkldnn/threads
    kwargs = paddleocr_init_kwargs(OCRConfig(cpu_threads=4))
    init_params = inspect.signature(PaddleOCR.__init__).parameters
    common = {k: v for k, v in kwargs.items() if k not in init_params}
    assert set(common) == {"device", "enable_mkldnn", "cpu_threads"}
    parse_common_args(common, default_enable_hpi=False)      # raises on unknown/invalid args
    predict_params = inspect.signature(PaddleOCR.predict).parameters
    assert set(paddleocr_predict_kwargs(OCRConfig(text_det_limit_side_len=960,
                                                  text_det_limit_type="max"))) <= set(predict_params)


def test_paddlex_uses_project_model_cache_from_any_working_directory(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PADDLE_PDX_CACHE_HOME"}
    env["PYTHONPATH"] = str(ROOT)
    out = subprocess.run([sys.executable, "-c",
                          "import config; import paddlex.utils.cache as c; print(c.CACHE_DIR)"],
                         cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == str(ROOT / "models" / "paddlex")


@pytest.mark.skipif(not HAVE_MODELS, reason="PaddleOCR models not in models/paddlex (run "
                                             "check_environment.py --models)")
def test_real_paddleocr_reads_synthetic_bibs():
    from PIL import Image, ImageDraw, ImageFont
    engine = PaddleOCREngine(OCRConfig())
    for text in ("10258", "T1234", "UT001", "00123"):
        img = Image.new("RGB", (700, 400), (80, 90, 80))
        d = ImageDraw.Draw(img)
        d.rectangle([100, 80, 600, 320], fill=(255, 255, 255))
        try:
            font = ImageFont.truetype("arialbd.ttf", 120)
        except OSError:
            font = ImageFont.load_default(size=120)
        d.text((130, 130), text, fill=(0, 0, 0), font=font)
        lines = engine.read(img)
        assert any(text in ln.text.replace(" ", "").upper() for ln in lines), (text, lines)
        assert all(0 <= ln.box.x1 < ln.box.x2 <= 700 for ln in lines)
