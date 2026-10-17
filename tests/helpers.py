"""Helpers shared by the detection tests."""
from PIL import Image


# ---------------------------------------------------------------- detection helpers

class ColorBlobDetector:
    """
    Stand-in for YOLO in integration tests: each "person" is a solid colour
    rectangle, and detect() returns the bounding box of each colour found.
    This makes the expected positions exact, so coordinate mapping, EXIF
    orientation and crops can be checked to the pixel. (The real model is
    tested in tests/e2e.)
    """
    name = "colour-blob stand-in"

    def __init__(self, colors, tol=40, confidence=0.9):
        self.colors, self.tol, self.confidence = colors, tol, confidence

    def detect(self, image):
        from PIL import ImageChops
        from detection.person_detector import RawDetection
        bands = image.convert("RGB").split()
        out = []
        for color in self.colors:
            masks = [b.point(lambda v, c=c: 255 if abs(v - c) <= self.tol else 0)
                     for b, c in zip(bands, color)]
            mask = ImageChops.multiply(ImageChops.multiply(masks[0], masks[1]), masks[2])
            bbox = mask.getbbox()
            if bbox:
                out.append(RawDetection(*bbox, confidence=self.confidence))
        return out


def draw_people(path, size, people, fmt="JPEG", orientation=None, background=(90, 110, 90)):
    """Write a photo with solid-colour 'people': people = [(color, (x1, y1, x2, y2)), ...]
    in RAW (stored) pixel coordinates; `orientation` adds an EXIF orientation tag."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, background)
    for color, box in people:
        img.paste(color, box)
    kwargs = {"quality": 95} if fmt == "JPEG" else {}
    if orientation:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kwargs["exif"] = exif.tobytes()
    img.save(path, fmt, **kwargs)
    return path


def make_test_config(tmp_path, **detection_overrides):
    """AppConfig whose data/debug folders live in tmp_path."""
    from config import AppConfig, DetectionConfig, PathsConfig
    data = tmp_path / "data"
    cfg = AppConfig(paths=PathsConfig(data_dir=data, debug_dir=data / "debug",
                                      database_file=data / "t.db", models_dir=tmp_path / "models",
                                      thumbnail_cache=data / "thumbs",
                                      default_export_root=tmp_path / "Export"),
                    detection=DetectionConfig(**detection_overrides))
    cfg.validate()
    return cfg


class ColorTextOCR:
    """
    Stand-in for PaddleOCR: each "bib" is a solid colour rectangle; read()
    returns the configured (text, confidence) for every such colour found in
    the crop, with its box in crop pixels. Records every crop it receives.
    """
    name = "colour-text stand-in"

    def __init__(self, texts, tol=30, fail_on_sizes=()):
        self.texts = texts                    # {(r, g, b): (text, confidence)}
        self.tol = tol
        self.fail_on_sizes = set(fail_on_sizes)
        self.calls = []                       # crop sizes received

    def read(self, image):
        from PIL import ImageChops
        from detection.geometry import Box
        from ocr.bib_reader import OCRLine
        self.calls.append(image.size)
        if image.size in self.fail_on_sizes:
            raise RuntimeError("simulated OCR failure")
        bands = image.convert("RGB").split()
        lines = []
        for color, (text, conf) in self.texts.items():
            masks = [b.point(lambda v, c=c: 255 if abs(v - c) <= self.tol else 0)
                     for b, c in zip(bands, color)]
            bbox = ImageChops.multiply(ImageChops.multiply(masks[0], masks[1]), masks[2]).getbbox()
            if bbox:
                lines.append(OCRLine(text, conf, Box(*bbox)))
        return lines
