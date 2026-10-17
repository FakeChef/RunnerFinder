"""
OCR engines. Anything with read(image) -> list[OCRLine] works, so tests can
use a stand-in. Boxes are in the pixel coordinates of the image passed in
(the crop); ocr/photo_ocr.py converts them to original photo coordinates.

PaddleOCREngine settings (all from config.OCRConfig):
  - light PP-OCRv5 "mobile" models, English recogniser
  - device "cpu", enable_mkldnn=False (verified on the Windows CPU install)
  - document-orientation, unwarping and text-line-orientation models off
    (photos are not scanned documents)
  - every recognised line is returned (score threshold 0); confidence
    filtering happens later, so rejected readings remain visible
  - model cache: PADDLE_PDX_CACHE_HOME, set to models/paddlex in config.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from config import OCRConfig
from detection.geometry import Box


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    box: Box                 # in crop pixels


class OCREngine(Protocol):
    name: str

    def read(self, image: Image.Image) -> list[OCRLine]:
        ...


def paddleocr_init_kwargs(cfg: OCRConfig) -> dict:
    """Constructor arguments for paddleocr.PaddleOCR."""
    kwargs = {
        "text_detection_model_name": cfg.text_detection_model,
        "text_recognition_model_name": cfg.text_recognition_model,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "device": cfg.device,
        "enable_mkldnn": cfg.enable_mkldnn,
    }
    if cfg.cpu_threads:
        kwargs["cpu_threads"] = cfg.cpu_threads
    return kwargs


def paddleocr_predict_kwargs(cfg: OCRConfig) -> dict:
    kwargs = {"text_rec_score_thresh": 0.0}   # keep every reading as evidence
    if cfg.text_det_limit_side_len:
        kwargs["text_det_limit_side_len"] = cfg.text_det_limit_side_len
    if cfg.text_det_limit_type:
        kwargs["text_det_limit_type"] = cfg.text_det_limit_type
    return kwargs


def _box_from_points(points) -> Box:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return Box(min(xs), min(ys), max(xs), max(ys))


def parse_paddle_results(results) -> list[OCRLine]:
    """
    Convert PaddleOCR 3.x predict() results (dict-like, keys "rec_texts",
    "rec_scores", "rec_boxes" [x1, y1, x2, y2], "rec_polys") to OCRLine.
    """
    lines = []
    for res in results or []:
        texts = list(res.get("rec_texts", []) or [])
        scores = list(res.get("rec_scores", []) or [])
        boxes = res.get("rec_boxes", None)
        boxes = [] if boxes is None else [list(b) for b in boxes]
        polys = res.get("rec_polys", None)
        polys = [] if polys is None else list(polys)
        for i, text in enumerate(texts):
            if i < len(boxes) and len(boxes[i]) == 4:
                box = Box(*(float(v) for v in boxes[i]))
            elif i < len(polys):
                box = _box_from_points(polys[i])
            else:
                continue
            score = float(scores[i]) if i < len(scores) else 0.0
            lines.append(OCRLine(str(text), score, box))
    return lines


class PaddleOCREngine:
    """
    The OCR device is decided once, here (utils/gpu_runtime.py):
      device          what PaddleOCR actually runs on ("cpu" or "gpu:0")
      device_info     GPU name, memory, CUDA/cuDNN versions (empty on CPU)
      device_note     why "auto" fell back to CPU, if it did
    An explicitly requested GPU that cannot be used raises GPUUnavailable.
    """

    def __init__(self, cfg: OCRConfig):
        import config  # noqa: F401  (sets PADDLE_PDX_CACHE_HOME, registers CUDA DLLs)
        from dataclasses import replace

        from utils.gpu_runtime import GPUUnavailable, active_paddle_device, resolve_ocr_device

        self.device, self.device_info, self.device_note = resolve_ocr_device(cfg.device)
        from paddleocr import PaddleOCR

        self.cfg = replace(cfg, device=self.device)
        self.ocr = PaddleOCR(**paddleocr_init_kwargs(self.cfg))
        if self.device.startswith("gpu"):
            active = active_paddle_device()
            if not (active or "").startswith("gpu"):
                raise GPUUnavailable(f"PaddleOCR loaded but Paddle reports device {active!r}, "
                                     "not the GPU (silent CPU fallback refused)")
        self.name = (f"PaddleOCR {cfg.text_detection_model} + {cfg.text_recognition_model} "
                     f"(device {self.device}, mkldnn {cfg.enable_mkldnn})")

    def read(self, image: Image.Image) -> list[OCRLine]:
        import numpy as np
        # PaddleOCR treats numpy input as BGR (OpenCV order).
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        return parse_paddle_results(self.ocr.predict(bgr, **paddleocr_predict_kwargs(self.cfg)))
