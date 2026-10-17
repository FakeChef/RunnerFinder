"""
Stage 1 - find the image regions that will be sent to OCR.

    image (EXIF-oriented, full resolution)
      -> downscaled copy (longest side = detection_image_size)
      -> person detector (YOLO26n)                    boxes in copy pixels
      -> map boxes back to full resolution            original pixels
      -> drop people that are too small
      -> expanded body crop per person                chest .. upper legs
      -> no person kept: full-image tiles (fallback)

All Region coordinates are in the full-resolution, EXIF-oriented image.
Crops are always cut from that full-resolution image, never from the
downscaled copy, so OCR gets every available pixel of the bib.

Interface for Phase 5 (OCR):
    result = finder.find(image)
    for region in result.regions:
        crop = crop_region(image, region)
        ... OCR on crop; boxes found in the crop are converted with
            geometry.crop_to_original(region.box, box_in_crop)
    if result.full_image_fallback_pending and no valid bib was read:
        for region in finder.full_image_regions(*image.size): ...
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from PIL import Image

from config import DetectionConfig
from detection.geometry import (Box, body_crop, center_crop_box, detection_size,
                                full_image_tiles, map_to_original)
from detection.person_detector import PersonDetector
from detection.ranking import rank_persons


@dataclass(frozen=True)
class PersonBox:
    index: int                    # P1, P2, ... left to right
    box: Box                      # original pixels
    confidence: float
    kept: bool
    skip_reason: str | None = None
    rank_score: float = 0.0


@dataclass(frozen=True)
class Region:
    """A crop to send to OCR, in original (EXIF-oriented) image pixels."""
    index: int
    box: Box                      # integer pixel bounds
    source: str                   # "person_body" | "center_crop" | "full_image"
    det_confidence: float         # person confidence; 0 for full-image tiles
    person_index: int | None = None
    notes: tuple[str, ...] = ()

    def xywh(self) -> tuple[int, int, int, int]:
        return self.box.xywh()


@dataclass
class DetectionResult:
    image_size: tuple[int, int]
    detection_size: tuple[int, int]
    persons: list[PersonBox]
    regions: list[Region]
    # People were found, so full-image OCR was not done now; with the
    # "no_valid_bib" setting, Phase 5 runs it if their crops give no bib.
    full_image_fallback_pending: bool
    fallback_reason: str | None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def kept_persons(self) -> list[PersonBox]:
        return [p for p in self.persons if p.kept]


class RegionFinder:
    def __init__(self, cfg: DetectionConfig, detector: PersonDetector | None,
                 mode: str | None = None):
        if cfg.strategy == "bib_model":
            raise NotImplementedError("The 'bib_model' strategy is reserved for a later version.")
        self.cfg = cfg
        self.detector = detector
        self.mode = mode or cfg.processing_mode
        if self.mode != "center" and detector is None:
            raise ValueError("a detector is required for non-center processing modes")

    def find(self, image: Image.Image) -> DetectionResult:
        cfg = self.cfg
        img_w, img_h = image.size
        timings = {}

        if self.mode == "center":
            box = center_crop_box(img_w, img_h, cfg.center_crop_width, cfg.center_crop_height)
            timings["resize_ms"] = 0.0
            timings["detect_ms"] = 0.0
            timings["regions_ms"] = 0.0
            return DetectionResult((img_w, img_h), (img_w, img_h), [],
                                   [Region(1, box, "center_crop", 0.0, None,
                                           ("center crop; YOLO bypassed",))],
                                   False, None, timings)

        t = time.perf_counter()
        det_w, det_h = detection_size(img_w, img_h, cfg.detection_image_size)
        small = image if (det_w, det_h) == (img_w, img_h) else \
            image.resize((det_w, det_h), Image.Resampling.BILINEAR, reducing_gap=2.0)
        timings["resize_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        raw = self.detector.detect(small)
        timings["detect_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        mapped = sorted(((map_to_original(Box(d.x1, d.y1, d.x2, d.y2), (det_w, det_h),
                                          (img_w, img_h)), d.confidence) for d in raw),
                        key=lambda bc: (bc[0].x1, bc[0].y1))
        persons = []
        for i, (box, conf) in enumerate(mapped, start=1):
            longest = max(box.width, box.height)
            if longest < cfg.min_person_size_px:
                persons.append(PersonBox(i, box, conf, False,
                                         f"too small ({longest:.0f} px < {cfg.min_person_size_px})"))
            else:
                persons.append(PersonBox(i, box, conf, True))

        top, bottom, side = cfg.crop_fractions()
        if self.mode in ("lite1", "lite2", "lite3"):
            t = time.perf_counter()
            ranked = rank_persons(persons, (img_w, img_h), cfg.rank_area_weight,
                                  cfg.rank_center_weight, cfg.rank_confidence_weight)
            scored = {p.index: score for p, score in ranked}
            persons = [PersonBox(p.index, p.box, p.confidence, p.kept, p.skip_reason,
                                 scored.get(p.index, 0.0)) for p in persons]
            ranked = [(next(p for p in persons if p.index == original.index), score)
                      for original, score in ranked]
            top_k = int(self.mode[-1])
            selected = ranked[:top_k]
            regions = []
            for p, score in selected:
                c = body_crop(p.box, img_w, img_h, top, bottom, side,
                              cfg.extend_crop_at_image_edges)
                regions.append(Region(len(regions) + 1, c.box, "person_body",
                                      p.confidence, p.index,
                                      c.notes + (f"rank_score={score:.4f}",)))
            timings["rank_ms"] = (time.perf_counter() - t) * 1000
            timings["regions_ms"] = 0.0
            return DetectionResult((img_w, img_h), (det_w, det_h), persons, regions,
                                   False, None if regions else "no person selected", timings)

        regions = []
        for p in persons:
            if p.kept:
                c = body_crop(p.box, img_w, img_h, top, bottom, side,
                              cfg.extend_crop_at_image_edges)
                regions.append(Region(len(regions) + 1, c.box, "person_body",
                                      p.confidence, p.index, c.notes))

        pending, reason = False, None
        if not regions:
            if cfg.full_image_fallback in ("no_person", "no_valid_bib"):
                reason = "no person kept" if persons else "no person detected"
                regions = self.full_image_regions(img_w, img_h)
            else:
                reason = "no person detected; full-image fallback is off"
        elif cfg.full_image_fallback == "no_valid_bib":
            pending = True
        timings["regions_ms"] = (time.perf_counter() - t) * 1000

        return DetectionResult((img_w, img_h), (det_w, det_h), persons, regions,
                               pending, reason, timings)

    def full_image_regions(self, img_w: int, img_h: int) -> list[Region]:
        tiles = full_image_tiles(img_w, img_h, self.cfg.full_image_tile_size,
                                 self.cfg.full_image_tile_overlap)
        note = (f"tile {{}} of {len(tiles)}",) if len(tiles) > 1 else ("whole image",)
        return [Region(i, b, "full_image", 0.0, None, (note[0].format(i),))
                for i, b in enumerate(tiles, start=1)]


def crop_region(image: Image.Image, region: Region) -> Image.Image:
    """The exact full-resolution pixels of a region (what OCR will receive)."""
    return image.crop(region.box.pil())
