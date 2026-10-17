"""
Central configuration for Runner Photo Finder.

All tunable settings live here. Edit the defaults below, or override
individual values from code:

    from config import get_config
    cfg = get_config()
    cfg.bib.min_length = 4

This module must stay lightweight: it is imported by the scanner and
database tests, which run without the AI packages installed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Stop Ultralytics from auto-installing packages at runtime: that would
# change the locked environment (and could break the shared OpenCV files).
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
# Skip PaddleOCR's slow "can I reach the model server?" check on every start.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Save Pillow's own Image.open before Ultralytics can replace it
# (see utils/pil_compat.py). Must stay above any Ultralytics import.
import utils.pil_compat  # noqa: E402,F401

# GPU environment only: make the NVIDIA CUDA/cuDNN wheel DLLs findable before
# PaddlePaddle is imported (utils/gpu_runtime.py). Does nothing in the CPU .venv.
from utils.gpu_runtime import register_cuda_dlls  # noqa: E402
register_cuda_dlls()

PROJECT_ROOT = Path(__file__).resolve().parent

# Keep Ultralytics off the network. Once the model file is on disk, run it
# fully offline: no connectivity probes on import and no usage analytics.
# (The very first run still needs the network once to download the model.)
# Analytics are additionally switched off in detection/person_detector.py.
if (PROJECT_ROOT / "models" / "yolo26n.pt").exists():
    os.environ.setdefault("YOLO_OFFLINE", "1")

# PaddleOCR (PaddleX) model cache: always inside this project, as an
# ABSOLUTE path, so it is the same whatever folder a command is started
# from. PaddleX reads this once, when it is first imported, so it must be
# set here, before any PaddleOCR import. Models end up in
# models/paddlex/official_models/. An existing environment variable wins.
PADDLE_CACHE_DIR = PROJECT_ROOT / "models" / "paddlex"
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLE_CACHE_DIR))


@dataclass
class BibFormatConfig:
    """Rules that decide whether an OCR string can be a race bib.

    Bibs are always handled as TEXT, never as numbers, so leading zeros
    ("UT001", "00123") are preserved everywhere.
    """
    min_length: int = 3
    max_length: int = 8
    allowed_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    require_digit: bool = True      # rejects sponsor words like "NIKE"
    max_letters: int = 3            # rejects longer words containing a digit
    # Optional race-specific regular expressions, e.g. [r"^\d{5}$", r"^UT\d{3}$"]
    patterns: list[str] = field(default_factory=list)


DETECTION_STRATEGIES = ("person_body", "bib_model")
FALLBACK_MODES = ("never", "no_person", "no_valid_bib")
PROCESSING_MODES = ("center", "lite1", "lite2", "lite3", "full")

# Body-crop presets: (top, bottom, side_margin) as fractions of the person
# box. top/bottom are measured from the top of the box; side_margin is added
# on each side. Bibs can be on the chest, abdomen, waist or thigh, so the
# default does NOT stop at the chest.
CROP_PRESETS: dict[str, tuple[float, float, float]] = {
    # chest, abdomen, waist and upper legs (default; trail and road races)
    "body": (0.10, 0.90, 0.15),
    # the whole person box: most recall, largest crops, slowest OCR
    "full_person": (0.00, 1.00, 0.10),
    # chest and abdomen only: smaller crops for road races with chest bibs
    "torso": (0.15, 0.65, 0.10),
}


@dataclass
class DetectionConfig:
    """Stage 1: decide which image regions are sent to OCR."""

    # "person_body": pretrained YOLO finds people, then an expanded body
    #                crop of each person is sent to OCR.
    # "bib_model":   reserved for a future dedicated bib detector.
    strategy: str = "person_body"
    # OCR processing policy. ``full`` preserves the Phase 5 behaviour.
    processing_mode: str = "full"
    lite_top_k: int = 2
    center_crop_width: float = 0.60
    center_crop_height: float = 0.60
    rank_area_weight: float = 0.45
    rank_center_weight: float = 0.35
    rank_confidence_weight: float = 0.20
    model_name: str = "yolo26n.pt"            # in models/, downloaded once
    bib_model_name: str = "bib_detector.pt"   # reserved
    device: str = "cpu"                       # "cpu", "auto" or e.g. "cuda:0"

    # Person detection. A low threshold keeps partly visible and blurred
    # runners; the debug images show every box with its confidence.
    person_conf_threshold: float = 0.25
    # Ignore people whose box is smaller than this on its LONGER side, in
    # original pixels (a cut-off runner can be short but wide).
    min_person_size_px: int = 120
    # The photo is downscaled so its longer side is at most this before
    # detection. Boxes are mapped back to full resolution; crops are always
    # cut from the full-resolution image.
    detection_image_size: int = 1280

    # Crop strategy: a name from CROP_PRESETS, or "custom" to use the three
    # values below.
    crop_strategy: str = "body"
    crop_top: float = 0.10
    crop_bottom: float = 0.90
    crop_side_margin: float = 0.15
    # If a person box touches the image edge, the body continues outside
    # the photo (partly visible runner). Then the crop runs to that edge
    # instead of being shortened, e.g. a runner cut off at the knees keeps
    # the waist and thighs in the crop.
    extend_crop_at_image_edges: bool = True

    # Full-image OCR fallback:
    #   "never"        - only OCR the body crops
    #   "no_person"    - OCR the whole image when no person was kept
    #   "no_valid_bib" - also when people were found but their crops gave
    #                    no valid bib (decided after OCR, in Phase 5)
    full_image_fallback: str = "no_valid_bib"
    # Full-image results need a higher OCR confidence, and must lie inside
    # a detected person when there is one, so banners and signs are ignored.
    full_image_min_ocr_confidence: float = 0.80
    full_image_require_person_overlap: bool = True
    # Large photos are split into overlapping tiles for full-image OCR, so
    # small bibs are not lost by downscaling a 24 MP photo in one piece.
    full_image_tile_size: int = 2560
    full_image_tile_overlap: int = 256

    def crop_fractions(self) -> tuple[float, float, float]:
        """(top, bottom, side_margin) for the active crop strategy."""
        if self.crop_strategy == "custom":
            return self.crop_top, self.crop_bottom, self.crop_side_margin
        return CROP_PRESETS[self.crop_strategy]


@dataclass
class DebugConfig:
    """Debug images for acceptance testing (always under data/debug)."""
    overlay_max_side: int = 2000       # overview images are downscaled to this
    overlay_jpeg_quality: int = 90
    crop_jpeg_quality: int = 95        # crops are saved at full resolution


@dataclass
class OCRConfig:
    """Stage 2: read text inside the regions from stage 1 (PaddleOCR 3.x)."""
    text_detection_model: str = "PP-OCRv5_mobile_det"
    text_recognition_model: str = "en_PP-OCRv5_mobile_rec"
    # "cpu" (default, the verified Windows install), "gpu" / "gpu:0" (needs the
    # separate GPU environment; an error if the GPU cannot be used) or "auto"
    # (GPU if it works, otherwise CPU with the reason printed and stored).
    device: str = "cpu"
    # oneDNN (MKL-DNN) acceleration. False is the setting verified on the
    # photographer's Windows CPU install; PaddleOCR's own default is True.
    enable_mkldnn: bool = False
    cpu_threads: int | None = None            # None: PaddleOCR default
    # Text-detection input size; None keeps the model defaults.
    text_det_limit_side_len: int | None = None
    text_det_limit_type: str | None = None    # "min" or "max"
    # Readings below this OCR confidence are kept as evidence but rejected.
    min_ocr_confidence: float = 0.50
    # Crops smaller than this (either side, pixels) are not sent to OCR.
    min_crop_side_px: int = 16
    # Fix common OCR letter/digit confusions in the numeric part of a bib
    # (O->0, I/L->1, S->5, B->8, Z->2, G->6, Q/D->0). Logged per reading.
    fix_digit_confusions: bool = True


@dataclass
class PathsConfig:
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    database_file: Path = PROJECT_ROOT / "data" / "runner_photo_finder.db"
    thumbnail_cache: Path = PROJECT_ROOT / "data" / "thumbnails"
    bib_rules_file: Path = PROJECT_ROOT / "config" / "bib_rules.yaml"   # noise filter (V0.3.1)
    default_export_root: Path = PROJECT_ROOT / "Exports"   # not "Export": on Windows that is the export/ package
    debug_dir: Path = PROJECT_ROOT / "data" / "debug"


@dataclass
class ThumbnailConfig:
    """Small previews for the Search page (utils/thumbnails.py), stored in
    paths.thumbnail_cache. Originals are never modified."""
    long_side_px: int = 512
    jpeg_quality: int = 80
    generate_during_analyze: bool = True   # from the image already decoded for OCR


@dataclass
class ScanConfig:
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")   # case-insensitive
    recursive: bool = True
    # Folders never scanned (in addition to hidden folders and the export folder).
    skip_dir_names: tuple[str, ...] = ("$RECYCLE.BIN", "System Volume Information")
    # Bytes read from the start and end of a file to build its fingerprint,
    # which recognises a photo after it was moved or renamed.
    fingerprint_chunk_bytes: int = 64 * 1024
    commit_every: int = 200    # database commit interval (resume granularity)


@dataclass
class AppConfig:
    bib: BibFormatConfig = field(default_factory=BibFormatConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    thumbnails: ThumbnailConfig = field(default_factory=ThumbnailConfig)

    def validate(self) -> None:
        """Raise ValueError for settings that make no sense."""
        b, d = self.bib, self.detection
        if not (1 <= b.min_length <= b.max_length):
            raise ValueError("bib.min_length must be >= 1 and <= bib.max_length")
        if d.strategy not in DETECTION_STRATEGIES:
            raise ValueError(f"detection.strategy must be one of {DETECTION_STRATEGIES}")
        dev = self.ocr.device.strip().lower()
        if not (dev in ("cpu", "gpu", "auto") or (dev.startswith("gpu:") and dev[4:].isdigit())):
            raise ValueError('ocr.device must be "cpu", "gpu", "gpu:N" or "auto"')
        if d.processing_mode not in PROCESSING_MODES:
            raise ValueError(f"detection.processing_mode must be one of {PROCESSING_MODES}")
        if d.lite_top_k not in (1, 2, 3):
            raise ValueError("detection.lite_top_k must be 1, 2 or 3")
        if not (0 < d.center_crop_width <= 1 and 0 < d.center_crop_height <= 1):
            raise ValueError("center crop fractions must be in (0, 1]")
        weights = (d.rank_area_weight, d.rank_center_weight, d.rank_confidence_weight)
        if any(w < 0 for w in weights) or not abs(sum(weights) - 1.0) < 1e-9:
            raise ValueError("runner ranking weights must be non-negative and sum to 1")
        if d.full_image_fallback not in FALLBACK_MODES:
            raise ValueError(f"detection.full_image_fallback must be one of {FALLBACK_MODES}")
        if d.crop_strategy not in (*CROP_PRESETS, "custom"):
            raise ValueError(f"detection.crop_strategy must be one of {(*CROP_PRESETS, 'custom')}")
        top, bottom, side = d.crop_fractions()
        if not (0 <= top < bottom <= 1) or side < 0:
            raise ValueError("crop fractions need 0 <= top < bottom <= 1 and side_margin >= 0")
        if not (0 < d.person_conf_threshold < 1):
            raise ValueError("detection.person_conf_threshold must be between 0 and 1")
        if d.detection_image_size < 64 or d.full_image_tile_size < 256:
            raise ValueError("detection_image_size >= 64 and full_image_tile_size >= 256 required")
        if not (0 <= d.full_image_tile_overlap < d.full_image_tile_size // 2):
            raise ValueError("full_image_tile_overlap must be smaller than half the tile size")


_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Return the shared configuration object (created on first use)."""
    global _config
    if _config is None:
        _config = AppConfig()
        _config.validate()
        _config.paths.data_dir.mkdir(parents=True, exist_ok=True)
    return _config
