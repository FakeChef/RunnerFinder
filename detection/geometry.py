"""
Box geometry for the detection stage. Pure Python, no image libraries.

COORDINATE SYSTEMS
  original:   the full-resolution photo AFTER EXIF orientation.
              Everything stored or passed on uses this system.
  detection:  the downscaled copy given to YOLO.
  crop:       pixels inside one crop (what OCR sees in Phase 5).

Boxes are (x1, y1, x2, y2) with x2/y2 EXCLUSIVE, so width = x2 - x1.
When a box is turned into pixel bounds it is rounded OUTWARDS (floor of
the start, ceil of the end): rounding never cuts pixels off a crop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def clamp(self, img_w: int, img_h: int) -> "Box":
        return Box(min(max(self.x1, 0), img_w), min(max(self.y1, 0), img_h),
                   min(max(self.x2, 0), img_w), min(max(self.y2, 0), img_h))

    def to_pixels(self) -> "Box":
        """Integer pixel bounds, rounded outwards."""
        return Box(math.floor(self.x1), math.floor(self.y1),
                   math.ceil(self.x2), math.ceil(self.y2))

    def xywh(self) -> tuple[int, int, int, int]:
        b = self.to_pixels()
        return int(b.x1), int(b.y1), int(b.width), int(b.height)

    def pil(self) -> tuple[int, int, int, int]:
        """(left, upper, right, lower) for PIL Image.crop()."""
        b = self.to_pixels()
        return int(b.x1), int(b.y1), int(b.x2), int(b.y2)


def detection_size(img_w: int, img_h: int, max_side: int) -> tuple[int, int]:
    """Size of the downscaled detection copy. Never upscales; keeps aspect ratio."""
    longest = max(img_w, img_h)
    if longest <= max_side:
        return img_w, img_h
    s = max_side / longest
    return max(1, round(img_w * s)), max(1, round(img_h * s))


def map_to_original(box: Box, det_size: tuple[int, int],
                    orig_size: tuple[int, int]) -> Box:
    """
    Map a box from the detection copy back to original pixels. Separate
    x and y factors are used because rounding the copy's size makes them
    differ slightly. The result is clamped to the original image.
    """
    sx = orig_size[0] / det_size[0]
    sy = orig_size[1] / det_size[1]
    return Box(box.x1 * sx, box.y1 * sy, box.x2 * sx, box.y2 * sy).clamp(*orig_size)


def edge_tolerance(length: int) -> float:
    """How close (in pixels) a box must be to an image edge to 'touch' it."""
    return max(2.0, 0.005 * length)


@dataclass(frozen=True)
class CropResult:
    box: Box                       # integer pixel bounds in original coordinates
    notes: tuple[str, ...]         # e.g. ("extended to bottom edge",)


def body_crop(person: Box, img_w: int, img_h: int, top: float, bottom: float,
              side_margin: float, extend_at_edges: bool = True) -> CropResult:
    """
    Expanded body crop for one person box.

    Vertically the crop runs from `top` to `bottom` (fractions of the box
    height, from the box top), horizontally the box is widened by
    `side_margin` x box width on each side (arms, side views, loose bibs).

    Partly visible runners: if the box touches the top or bottom edge of
    the photo, the person continues outside it, so the fractions would be
    measured against the wrong height. The crop then extends to that edge.
    """
    notes = []
    bw, bh = person.width, person.height
    y1 = person.y1 + top * bh
    y2 = person.y1 + bottom * bh
    if extend_at_edges:
        if person.y1 <= edge_tolerance(img_h):
            y1 = 0.0
            notes.append("touches top edge: crop starts at top")
        if person.y2 >= img_h - edge_tolerance(img_h):
            y2 = float(img_h)
            notes.append("touches bottom edge: crop extended to bottom")
    x1 = person.x1 - side_margin * bw
    x2 = person.x2 + side_margin * bw
    box = Box(x1, y1, x2, y2).clamp(img_w, img_h).to_pixels().clamp(img_w, img_h)
    if box.x1 == 0 or box.x2 == img_w:
        if person.x1 <= edge_tolerance(img_w) or person.x2 >= img_w - edge_tolerance(img_w):
            notes.append("touches side edge")
    return CropResult(box, tuple(notes))


def center_crop_box(img_w: int, img_h: int, width_fraction: float = 0.60,
                    height_fraction: float = 0.60) -> Box:
    """Return a centered crop box, preserving the original pixel grid."""
    if not (0 < width_fraction <= 1 and 0 < height_fraction <= 1):
        raise ValueError("center crop fractions must be in (0, 1]")
    w = max(1, round(img_w * width_fraction))
    h = max(1, round(img_h * height_fraction))
    x1 = (img_w - w) // 2
    y1 = (img_h - h) // 2
    return Box(x1, y1, x1 + w, y1 + h)


def full_image_tiles(img_w: int, img_h: int, tile: int, overlap: int) -> list[Box]:
    """
    Overlapping tiles that together cover every pixel. A photo that fits in
    one tile gives a single box covering the whole photo.
    """
    def starts(length: int) -> list[int]:
        if length <= tile:
            return [0]
        n = math.ceil((length - overlap) / (tile - overlap))
        step = (length - tile) / (n - 1)
        return [round(i * step) for i in range(n)]

    return [Box(x, y, min(x + tile, img_w), min(y + tile, img_h))
            for y in starts(img_h) for x in starts(img_w)]


def crop_to_original(crop: Box, inner: Box) -> Box:
    """Convert a box found inside a crop (e.g. by OCR) to original pixels."""
    return Box(crop.x1 + inner.x1, crop.y1 + inner.y1, crop.x1 + inner.x2, crop.y1 + inner.y2)


def fraction_inside(inner: Box, outer: Box) -> float:
    """Share of `inner`'s area that lies inside `outer` (0..1)."""
    if inner.area == 0:
        return 0.0
    ix = max(0.0, min(inner.x2, outer.x2) - max(inner.x1, outer.x1))
    iy = max(0.0, min(inner.y2, outer.y2) - max(inner.y1, outer.y1))
    return (ix * iy) / inner.area


def inside_any(box: Box, others: list[Box], min_fraction: float = 0.5) -> bool:
    """True if at least `min_fraction` of `box` lies inside one of `others`."""
    return any(fraction_inside(box, o) >= min_fraction for o in others)


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0
