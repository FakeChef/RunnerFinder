"""
Per-photo OCR: regions from the detection stage -> OCR -> bib candidates
-> one deduplicated list of bibs PER PHOTOGRAPH.

Rules (Phase 4 review):
  * Results belong to the photograph. A bib read inside a person's crop is
    NOT assigned to that person: crops overlap in dense groups, so the text
    may belong to a neighbour. The crop and its person are kept only as
    evidence (Reading.region_id / region_person_index).
  * The same normalised bib is reported once per photograph (PhotoBib),
    however many crops or tiles read it. Every individual reading, accepted
    or rejected, stays in PhotoOCRResult.readings with its raw text,
    confidence, coordinates, source crop and rejection reason.
  * OCR always runs on full-resolution crops.
  * "No valid bib read" never means "no bib in the photo": region statuses
    say what happened (no text found, text but no valid bib, low confidence,
    crop too small, OCR error).

All coordinates are original-photo pixels after EXIF orientation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from PIL import Image

from config import AppConfig
from detection.bib_detector import DetectionResult, Region, RegionFinder, crop_region
from detection.geometry import Box, crop_to_original, inside_any
from ocr.bib_reader import OCREngine
from ocr.candidates import candidates_from_text
from ocr.numeric_candidates import numeric_identifier

# Reading rejection reasons added at this stage (format reasons come from
# ocr/bib_format.py and ocr/candidates.py).
LOW_OCR_CONFIDENCE = "low_ocr_confidence"
FULL_IMAGE_LOW_CONFIDENCE = "full_image_low_confidence"
FULL_IMAGE_OUTSIDE_PERSON = "full_image_outside_person"

# Region statuses
BIB_READ = "bib_read"
TEXT_NO_VALID_BIB = "text_but_no_valid_bib"
NO_TEXT = "no_text_detected"
TOO_SMALL = "crop_too_small_for_ocr"
OCR_ERROR = "ocr_error"


def region_id(region: Region) -> str:
    prefix = {"person_body": "C", "center_crop": "X", "full_image": "T"}[region.source]
    return f"{prefix}{region.index}"


@dataclass(frozen=True)
class Reading:
    """One candidate from one OCR line in one region. Raw evidence."""
    region_id: str
    region_source: str
    region_person_index: int | None     # whose crop it was read in (NOT ownership)
    line_index: int
    raw_text: str
    ocr_confidence: float
    line_box: Box                       # original pixels
    token: str
    normalized: str
    bib: str | None
    corrected: bool
    accepted: bool
    reason: str | None                  # None when accepted
    numeric_identifier: str | None = None


@dataclass
class RegionOCR:
    region: Region
    region_id: str
    status: str
    lines: int = 0
    ocr_ms: float = 0.0
    error: str = ""


@dataclass
class PhotoBib:
    """One bib for one photograph (deduplicated)."""
    bib: str
    best_confidence: float
    readings: int                        # accepted readings merged into this bib
    region_ids: tuple[str, ...]
    best_box: Box                        # line box of the best reading (original pixels)
    corrected: bool                      # best reading needed a confusion fix
    numeric_identifier: str | None = None


@dataclass
class PhotoOCRResult:
    detection: DetectionResult
    regions: list[RegionOCR]
    readings: list[Reading]
    bibs: list[PhotoBib]
    fallback_used: str | None            # None, "no_person" or "no_valid_bib_in_crops"
    timings_ms: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.bibs:
            return "bib_read"
        counts = {}
        for r in self.regions:
            counts[r.status] = counts.get(r.status, 0) + 1
        detail = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no regions"
        return f"no_valid_bib_read ({detail})"


def deduplicate(readings: list[Reading]) -> list[PhotoBib]:
    """One PhotoBib per accepted normalised bib, best confidence first."""
    groups: dict[str, list[Reading]] = {}
    for r in readings:
        if r.accepted and r.bib:
            groups.setdefault(r.bib, []).append(r)
    bibs = []
    for bib, rs in groups.items():
        best = max(rs, key=lambda r: r.ocr_confidence)
        bibs.append(PhotoBib(bib, best.ocr_confidence, len(rs),
                             tuple(sorted({r.region_id for r in rs})), best.line_box,
                             best.corrected, best.numeric_identifier))
    return sorted(bibs, key=lambda b: (-b.best_confidence, b.bib))


class PhotoOCR:
    def __init__(self, cfg: AppConfig, finder: RegionFinder, engine: OCREngine,
                 registered: set[str] | None = None):
        self.cfg = cfg
        self.finder = finder
        self.engine = engine
        self.registered = registered

    def process(self, image: Image.Image, detection: DetectionResult | None = None) -> PhotoOCRResult:
        # detection (V1.3 three-stage pipeline): self.finder.find(image), already run in the detection thread
        det = self.finder.find(image) if detection is None else detection
        person_boxes = [p.box for p in det.persons]
        regions, readings, timings = [], [], {"ocr_ms": 0.0, "fallback_ocr_ms": 0.0}

        t = time.perf_counter()
        for region in det.regions:
            self._ocr_region(image, region, person_boxes, regions, readings)
        timings["ocr_ms"] = (time.perf_counter() - t) * 1000

        fallback = "no_person" if det.regions and det.regions[0].source == "full_image" else None
        if det.full_image_fallback_pending and not any(r.accepted for r in readings):
            fallback = "no_valid_bib_in_crops"
            t = time.perf_counter()
            for region in self.finder.full_image_regions(*image.size):
                self._ocr_region(image, region, person_boxes, regions, readings)
            timings["fallback_ocr_ms"] = (time.perf_counter() - t) * 1000

        return PhotoOCRResult(det, regions, readings, deduplicate(readings), fallback,
                              {**det.timings_ms, **timings})

    def _ocr_region(self, image, region, person_boxes, regions, readings) -> None:
        rid = region_id(region)
        if min(region.box.width, region.box.height) < self.cfg.ocr.min_crop_side_px:
            regions.append(RegionOCR(region, rid, TOO_SMALL))
            return
        crop = crop_region(image, region)              # full resolution
        t = time.perf_counter()
        try:
            lines = self.engine.read(crop)
        except Exception as exc:  # noqa: BLE001 - one failing crop must not stop the photo
            regions.append(RegionOCR(region, rid, OCR_ERROR, 0, (time.perf_counter() - t) * 1000,
                                     f"{type(exc).__name__}: {exc}"))
            return
        ocr_ms = (time.perf_counter() - t) * 1000

        accepted_any = False
        for li, line in enumerate(lines, start=1):
            box = crop_to_original(region.box, line.box)
            for c in candidates_from_text(line.text, self.cfg.bib,
                                          self.cfg.ocr.fix_digit_confusions, self.registered):
                reason = c.format_reason or self._confidence_reason(region, line.confidence,
                                                                     box, person_boxes)
                readings.append(Reading(rid, region.source, region.person_index, li, line.text,
                                        line.confidence, box, c.token, c.normalized, c.bib,
                                        c.corrected, reason is None, reason,
                                        numeric_identifier(c.bib or c.normalized or c.token)))
                accepted_any |= reason is None
        status = BIB_READ if accepted_any else (TEXT_NO_VALID_BIB if lines else NO_TEXT)
        regions.append(RegionOCR(region, rid, status, len(lines), ocr_ms))

    def _confidence_reason(self, region, confidence, box, person_boxes) -> str | None:
        ocr_cfg, det_cfg = self.cfg.ocr, self.cfg.detection
        if confidence < ocr_cfg.min_ocr_confidence:
            return LOW_OCR_CONFIDENCE
        if region.source == "full_image":
            if confidence < det_cfg.full_image_min_ocr_confidence:
                return FULL_IMAGE_LOW_CONFIDENCE
            if (det_cfg.full_image_require_person_overlap and person_boxes
                    and not inside_any(box, person_boxes)):
                return FULL_IMAGE_OUTSIDE_PERSON
        return None


def detection_rows(result: PhotoOCRResult) -> list[dict]:
    """
    Rows for the bib_detection table (written in Phase 6): one per accepted
    reading, plus one bib_number=NULL row per region where no valid bib was
    read. person_bbox_* stay NULL on purpose: a bib is not assigned to a
    person. Duplicate readings are kept; the v_photo_bibs view merges them.
    """
    rows = []
    for r in result.readings:
        if r.accepted:
            x, y, w, h = r.line_box.xywh()
            rows.append({"bib_number": r.bib, "confidence": r.ocr_confidence,
                         "det_confidence": _region_conf(result, r.region_id),
                         "ocr_confidence": r.ocr_confidence, "bbox_x": x, "bbox_y": y,
                         "bbox_width": w, "bbox_height": h, "ocr_text": r.raw_text,
                         "source": r.region_source})
    for reg in result.regions:
        if reg.status != BIB_READ:
            x, y, w, h = reg.region.xywh()
            rows.append({"bib_number": None, "confidence": None,
                         "det_confidence": reg.region.det_confidence, "ocr_confidence": None,
                         "bbox_x": x, "bbox_y": y, "bbox_width": w, "bbox_height": h,
                         "ocr_text": reg.status, "source": reg.region.source})
    return rows


MAX_EVIDENCE_READINGS = 10     # keeps evidence_json small; CSVs have everything


def photo_bib_rows(result: PhotoOCRResult, bib_filter=None) -> list[dict]:
    """
    Rows for bib_detection (schema 3): one per deduplicated bib of the photo,
    the same content as photo_bibs.csv, plus a small evidence_json with the
    accepted readings behind it. Rejected readings are not included.
    V0.3.1: evidence_json["filter"] holds the noise-filter result
    (ocr/bib_filter.py); every row is stored whatever the result.
    """
    from ocr.bib_filter import PhotoCandidate, evaluate_photo
    if bib_filter is None:
        from ocr.bib_filter import default_filter
        bib_filter = default_filter()
    # V1.3.1: all candidates of the photo are scored together (relative text size in a person crop)
    cands = []
    for b in result.bibs:
        accepted = [r for r in result.readings if r.accepted and r.bib == b.bib]
        best = max(accepted, key=lambda r: r.ocr_confidence) if accepted else None
        cands.append(PhotoCandidate(b.bib, b.numeric_identifier, b.best_confidence,
                                    tuple(r.raw_text for r in accepted),
                                    best.region_source if best else source_from_region_ids(b.region_ids),
                                    b.best_box.height, tuple(b.region_ids)))
    results = evaluate_photo(bib_filter, cands)
    rows = []
    for b, res in zip(result.bibs, results):
        accepted = sorted((r for r in result.readings if r.accepted and r.bib == b.bib),
                          key=lambda r: -r.ocr_confidence)
        best = accepted[0] if accepted else None
        x, y, w, h = b.best_box.xywh()
        evidence = {"readings": [
            {"raw_text": r.raw_text, "ocr_confidence": round(r.ocr_confidence, 4),
             "region_id": r.region_id, "region_source": r.region_source,
             "box": list(r.line_box.xywh()), "letter_digit_fix": r.corrected}
            for r in accepted[:MAX_EVIDENCE_READINGS]],
            "fallback_used": result.fallback_used}
        evidence["filter"] = res.as_dict()
        rows.append({
            "bib_number": b.bib, "numeric_bib": b.numeric_identifier,
            "confidence": b.best_confidence, "ocr_confidence": b.best_confidence,
            "det_confidence": _region_conf(result, best.region_id) if best else None,
            "bbox_x": x, "bbox_y": y, "bbox_width": w, "bbox_height": h,
            "ocr_text": best.raw_text if best else None,
            "source": best.region_source if best else source_from_region_ids(b.region_ids),
            "readings": b.readings, "region_ids": ";".join(b.region_ids),
            "letter_digit_fix": int(b.corrected),
            "evidence_json": json.dumps(evidence, ensure_ascii=False),
        })
    return rows


_REGION_PREFIX_SOURCE = {"C": "person_body", "X": "center_crop", "T": "full_image"}


def source_from_region_ids(region_ids) -> str:
    """'C1;C2' -> 'person_body' (first region id decides). Used for CSV imports."""
    ids = region_ids.split(";") if isinstance(region_ids, str) else list(region_ids)
    first = next((i.strip() for i in ids if i.strip()), "")
    return _REGION_PREFIX_SOURCE.get(first[:1], "person_body")


def _region_conf(result: PhotoOCRResult, rid: str) -> float | None:
    return next((r.region.det_confidence for r in result.regions if r.region_id == rid), None)
