"""
Debug images for acceptance testing.

For one photo this writes, into an output folder under data/debug/:
  <name>_overview.jpg    downscaled COPY of the photo with person boxes,
                         crop regions, confidences, timings and file name
  <name>_crop01_person_body.jpg, ...
                         the exact full-resolution pixels each region
                         sends to OCR
  <name>_detections.json all numbers, in original pixel coordinates

The original photo is only read, never written.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from detection.bib_detector import DetectionResult, crop_region

PERSON = (30, 120, 255)        # blue: person box from YOLO
CROP = (255, 140, 0)           # orange, dashed: region sent to OCR
SKIPPED = (150, 150, 150)      # grey: person ignored (too small)
TILE = (220, 0, 200)           # magenta: full-image tile (fallback)
HEADER_BG, HEADER_FG = (20, 20, 20), (255, 255, 255)

# Fonts with Chinese glyphs first, so file names like 太湖 render.
_FONT_CANDIDATES = ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "arial.ttf",
                    "NotoSansCJK-Regular.ttc", "DejaVuSans.ttf")


def _font(size: int) -> ImageFont.ImageFont:
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def ensure_inside(path: Path, root: Path) -> Path:
    """Refuse to write anywhere except below `root` (the data directory)."""
    path, root = Path(path).resolve(), Path(root).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Debug output must be inside {root}, got {path}")
    return path


def _dashed_rect(draw, box, color, width, dash):
    x1, y1, x2, y2 = box
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = max(abs(bx - ax), abs(by - ay))
        steps = max(1, int(length // dash))
        for i in range(0, steps, 2):
            s, e = i / steps, min(1, (i + 1) / steps)
            draw.line((ax + (bx - ax) * s, ay + (by - ay) * s,
                       ax + (bx - ax) * e, ay + (by - ay) * e), fill=color, width=width)


def _label(draw, xy, text, color, font, img_w, placed=None):
    """Draw a filled label above xy. If it would cover an earlier label
    (runners side by side), move it down until it is free."""
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    w, h = r - l + 8, b - t + 6
    x = min(max(0, x), max(0, img_w - w))
    y = max(0, y - h)
    if placed is not None:
        for _ in range(20):
            if not any(x < px2 and px1 < x + w and y < py2 and py1 < y + h
                       for px1, py1, px2, py2 in placed):
                break
            y += h + 2
        placed.append((x, y, x + w, y + h))
    draw.rectangle((x, y, x + w, y + h), fill=color)
    draw.text((x + 4, y + 3 - t), text, fill=(255, 255, 255), font=font)


BIB_OK = (0, 160, 60)          # green: accepted bib reading (Phase 5)
BIB_REJECTED = (215, 30, 30)   # red: OCR text with digits that was rejected


def render_overview(image: Image.Image, result: DetectionResult, header_lines: list[str],
                    max_side: int, marks: list | None = None) -> Image.Image:
    """marks: optional [(Box, label, color)] drawn on top (OCR readings)."""
    img_w, img_h = image.size
    s = min(1.0, max_side / max(img_w, img_h))
    view = image.resize((max(1, round(img_w * s)), max(1, round(img_h * s))),
                        Image.Resampling.BILINEAR) if s < 1 else image.copy()
    vw, vh = view.size
    lw = max(2, round(max(vw, vh) / 500))
    font = _font(max(12, round(max(vw, vh) / 70)))
    draw = ImageDraw.Draw(view)

    def sc(b):
        return (b.x1 * s, b.y1 * s, b.x2 * s - 1, b.y2 * s - 1)

    # Crops first, person boxes on top. Each body crop belongs to one person,
    # so its number goes into that person's label (no separate labels that
    # would collide when runners stand close together).
    crop_of = {r.person_index: r.index for r in result.regions if r.person_index is not None}
    placed: list[tuple] = []
    for r in result.regions:
        color = CROP if r.source == "person_body" else TILE
        _dashed_rect(draw, sc(r.box), color, lw + 1, dash=max(8, lw * 5))
        if r.source != "person_body":
            _label(draw, (r.box.x1 * s + lw, r.box.y1 * s + lw * 8), f"tile {r.index}", TILE, font, vw,
                   placed)
    for p in result.persons:
        color = PERSON if p.kept else SKIPPED
        draw.rectangle(sc(p.box), outline=color, width=lw if p.kept else max(1, lw // 2))
        text = f"P{p.index} {p.confidence:.2f}"
        text += f" / crop {crop_of[p.index]}" if p.index in crop_of else " too small"
        _label(draw, (p.box.x1 * s, p.box.y1 * s), text, color, font, vw, placed)
    for box, text, color in marks or []:
        draw.rectangle(sc(box), outline=color, width=lw + 1)
        _label(draw, (box.x1 * s, box.y2 * s + font.size + 10), text, color, font, vw, placed)

    # Header strip ABOVE the photo, so it never hides part of the image.
    # Long lines are wrapped at " | " so nothing is cut off on narrow photos.
    hfont = _font(max(13, round(vw / 75)))
    line_h = sum(hfont.getmetrics()) + 4
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    wrapped = []
    for line in header_lines:
        current = ""
        for part in line.split(" | "):
            trial = f"{current} | {part}" if current else part
            if current and measure.textlength(trial, font=hfont) > vw - 16:
                wrapped.append(current)
                current = "   " + part
            else:
                current = trial
        wrapped.append(current)
    header_lines = wrapped
    header = Image.new("RGB", (vw, line_h * len(header_lines) + 10), HEADER_BG)
    hd = ImageDraw.Draw(header)
    for i, line in enumerate(header_lines):
        hd.text((8, 5 + i * line_h), line, fill=HEADER_FG, font=hfont)
    canvas = Image.new("RGB", (vw, header.height + vh), HEADER_BG)
    canvas.paste(header, (0, 0))
    canvas.paste(view, (0, header.height))
    return canvas


def write_debug(out_dir: Path, data_root: Path, name: str, source_path: str,
                image: Image.Image, result: DetectionResult, timings_ms: dict,
                orientation: int | None, max_side: int = 2000,
                overlay_quality: int = 90, crop_quality: int = 95, time_note: str = "",
                marks: list | None = None, extra_header: list[str] | None = None,
                crop_regions: list | None = None) -> dict:
    """Write overview, crops and JSON. Returns the paths written."""
    out_dir = ensure_inside(out_dir, data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    if Path(source_path).resolve().parent == out_dir:
        raise ValueError("Refusing to write debug files next to the original photo")

    kept = len(result.kept_persons)
    skipped = len(result.persons) - kept
    w, h = result.image_size
    dw, dh = result.detection_size
    t = timings_ms
    fallback = ("full-image tiles used: " + result.fallback_reason) if result.fallback_reason \
        else ("full-image fallback: runs only if the crops give no valid bib"
              if result.full_image_fallback_pending else "full-image fallback: not needed")
    header = [
        f"File: {os.path.basename(source_path)}",
        f"Image {w}x{h} (EXIF orientation {orientation or 1}) | detection copy {dw}x{dh} | "
        f"persons kept {kept}, skipped {skipped} | OCR regions {len(result.regions)}",
        f"Time: decode {t.get('decode_ms', 0):.0f} ms | resize {t.get('resize_ms', 0):.0f} ms | "
        f"detect {t.get('detect_ms', 0):.0f} ms | crops {t.get('crop_ms', 0):.0f} ms | "
        f"total {t.get('total_ms', 0):.0f} ms{time_note}",
        f"{fallback}",
        "blue = person (confidence) | orange dashed = crop sent to OCR | "
        "grey = person too small | magenta dashed = full-image tile"
        + (" | green = bib read | red = rejected text" if marks is not None else ""),
        *(extra_header or []),
    ]
    overview_path = out_dir / f"{name}_overview.jpg"
    render_overview(image, result, header, max_side, marks).save(overview_path,
                                                                 quality=overlay_quality)

    crop_paths = []
    regions_to_save = crop_regions if crop_regions is not None else result.regions
    for r in regions_to_save:
        p = out_dir / f"{name}_crop{r.index:02d}_{r.source}.jpg"
        crop_region(image, r).save(p, quality=crop_quality)
        crop_paths.append(str(p))

    data = {
        "file": source_path,
        "coordinate_system": "original pixels after EXIF orientation; boxes are x, y, width, height",
        "image_size": [w, h], "detection_size": [dw, dh], "exif_orientation": orientation,
        "timings_ms": {k: round(v, 1) for k, v in timings_ms.items()},
        "persons": [{"index": p.index, "box": list(p.box.xywh()), "confidence": round(p.confidence, 4),
                     "kept": p.kept, "skip_reason": p.skip_reason} for p in result.persons],
        "regions": [{"index": r.index, "source": r.source, "box": list(r.xywh()),
                     "person_index": r.person_index, "det_confidence": round(r.det_confidence, 4),
                     "notes": list(r.notes), "crop_file": cp}
                    for r, cp in zip(regions_to_save, crop_paths)],
        "full_image_fallback_pending": result.full_image_fallback_pending,
        "fallback_reason": result.fallback_reason,
    }
    json_path = out_dir / f"{name}_detections.json"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"overview": str(overview_path), "crops": crop_paths, "json": str(json_path)}
