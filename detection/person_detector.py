"""
Person detectors. Anything with a detect(image) method returning
RawDetection boxes works, which lets tests use a simple stand-in instead
of the real model.

Boxes returned by detect() are in the pixel coordinates of the image that
was passed in (the downscaled detection copy). Mapping back to the
original photo happens in detection/bib_detector.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class RawDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


def disable_ultralytics_analytics() -> None:
    """
    Ultralytics sends anonymous usage events (model, task, speeds, CPU,
    versions; no images) to Google Analytics after predictions when its
    "sync" setting is on, which is the default. Switch it off with the
    documented settings API. This is stored in the user's Ultralytics
    settings file (Windows: %APPDATA%\\Ultralytics\\settings.json); undo with
    `yolo settings sync=True`. Must run before the first prediction.
    """
    from ultralytics import settings
    if settings.get("sync", False):
        settings.update({"sync": False})


class PersonDetector(Protocol):
    name: str

    def detect(self, image: Image.Image) -> list[RawDetection]:
        ...


class YoloPersonDetector:
    """
    Ultralytics YOLO (pretrained on COCO), restricted to class 0 = person.
    The weights file is downloaded into models/ on first use.
    """

    PERSON_CLASS = 0

    def __init__(self, weights: Path, conf: float = 0.25, imgsz: int = 1280,
                 device: str = "cpu"):
        import config  # noqa: F401  (sets YOLO_AUTOINSTALL=False before importing ultralytics)
        from ultralytics import YOLO

        disable_ultralytics_analytics()

        Path(weights).parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = None if device == "auto" else device   # None: Ultralytics picks
        self.name = f"{Path(weights).name} (conf {conf}, imgsz {imgsz}, device {device})"

    def detect(self, image: Image.Image) -> list[RawDetection]:
        # A PIL RGB image is passed directly; Ultralytics converts it itself.
        result = self.model.predict(image, classes=[self.PERSON_CLASS], conf=self.conf,
                                    imgsz=self.imgsz, device=self.device,
                                    max_det=300, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy().tolist()
        confs = boxes.conf.cpu().numpy().tolist()
        classes = boxes.cls.cpu().numpy().tolist()
        return [RawDetection(*b, confidence=float(c))
                for b, c, k in zip(xyxy, confs, classes) if int(k) == self.PERSON_CLASS]
