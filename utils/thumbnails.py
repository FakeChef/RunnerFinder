"""
Thumbnail cache for the Search page (V0.3.1).

    data/thumbnails/<first 2 characters of fingerprint>/<fingerprint>.jpg

* Keyed by the photo fingerprint the scanner already stores (photos.fingerprint),
  so a moved or renamed photo keeps its thumbnail and changed content gets a
  new one. Identical copies of a photo share one thumbnail.
* 512 px on the long side, JPEG quality 80 (config.ThumbnailConfig), already
  rotated according to EXIF and saved WITHOUT EXIF, so no viewer can rotate
  it a second time.
* Written to a temporary file in the same folder, then renamed: a crash never
  leaves a half-written thumbnail. Writes outside the cache folder are refused.
* Original photos are only ever opened for reading.
* Any failure returns None: the caller shows the original instead. A thumbnail
  problem never stops an analysis or a search.

Two ways to create one:
  save_from_image()   during "analyze", from the image already decoded and
                      rotated for OCR (no second decode)
  ensure_thumbnail()  lazily, e.g. on the Search page, with a fast reduced-size
                      JPEG decode ("draft" mode)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

from config import ThumbnailConfig
from utils.pil_compat import pillow_open


def thumbnail_path(cache_dir: Path | str, fingerprint: str) -> Path:
    fp = (fingerprint or "").strip().lower()
    if len(fp) < 8 or not all(ch in "0123456789abcdef" for ch in fp):
        raise ValueError(f"not a photo fingerprint: {fingerprint!r}")
    return Path(cache_dir) / fp[:2] / f"{fp}.jpg"


def existing_thumbnail(cache_dir: Path | str, fingerprint: str | None) -> Path | None:
    if not fingerprint:
        return None
    try:
        p = thumbnail_path(cache_dir, fingerprint)
    except ValueError:
        return None
    return p if p.is_file() else None


def _write(small: Image.Image, cache_dir: Path | str, fingerprint: str, quality: int) -> Path:
    cache = Path(cache_dir).resolve()
    target = thumbnail_path(cache, fingerprint).resolve()
    if not target.is_relative_to(cache):
        raise ValueError(f"refusing to write outside the thumbnail cache: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".jpg", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            small.convert("RGB").save(fh, "JPEG", quality=quality, optimize=True)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def _shrink(image: Image.Image, long_side: int) -> Image.Image:
    """Smaller copy, long side <= long_side; the input image is not changed."""
    factor = max(1, max(image.size) // (long_side * 2))
    small = image.reduce(factor) if factor > 1 else image.copy()   # fast box reduction first
    small.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    return small


def save_from_image(image: Image.Image, cache_dir: Path | str, fingerprint: str,
                    cfg: ThumbnailConfig | None = None, overwrite: bool = False) -> Path | None:
    """Thumbnail from an image that is already decoded AND rotated (the OCR image)."""
    cfg = cfg or ThumbnailConfig()
    try:
        if not overwrite and existing_thumbnail(cache_dir, fingerprint):
            return thumbnail_path(cache_dir, fingerprint)
        return _write(_shrink(image, cfg.long_side_px), cache_dir, fingerprint, cfg.jpeg_quality)
    except Exception:  # noqa: BLE001 - a thumbnail must never break the analysis
        return None


def ensure_thumbnail(photo_path: Path | str, cache_dir: Path | str, fingerprint: str | None,
                     cfg: ThumbnailConfig | None = None) -> Path | None:
    """Existing thumbnail, or a new one made from the original (read-only); None if impossible."""
    cfg = cfg or ThumbnailConfig()
    found = existing_thumbnail(cache_dir, fingerprint)
    if found or not fingerprint:
        return found
    try:
        with pillow_open()(photo_path) as im:
            if im.format == "JPEG":
                # decode at a reduced scale directly (much faster than full size)
                im.draft("RGB", (cfg.long_side_px * 2, cfg.long_side_px * 2))
            rotated = ImageOps.exif_transpose(im)
            return _write(_shrink(rotated, cfg.long_side_px), cache_dir, fingerprint, cfg.jpeg_quality)
    except Exception:  # noqa: BLE001 - missing/corrupt original: caller falls back
        return None


def display_path(photo_path: Path | str, cache_dir: Path | str, fingerprint: str | None,
                 cfg: ThumbnailConfig | None = None) -> Path:
    """What the Search page should show: thumbnail, else a new thumbnail, else the original."""
    return ensure_thumbnail(photo_path, cache_dir, fingerprint, cfg) or Path(photo_path)
