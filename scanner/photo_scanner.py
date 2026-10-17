"""
Photo scanner: find image files and read their metadata.

READ-ONLY GUARANTEE
    This module only lists directories, calls os.stat(), and opens files
    in binary READ mode ("rb", or Pillow's Image.open, which reads only).
    It never writes, renames, touches or re-saves an original photo.

COORDINATE SYSTEM
    load_image_oriented() returns the photo with its EXIF orientation
    applied, at full resolution. PhotoInfo.width/height describe that same
    oriented image. Every bounding box stored later uses this system.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image, ImageOps, UnidentifiedImageError

from utils.pil_compat import pillow_open

# EXIF tag numbers (see the EXIF 2.32 specification)
TAG_ORIENTATION = 0x0112
TAG_MAKE = 0x010F
TAG_MODEL = 0x0110
TAG_DATETIME = 0x0132              # IFD0: last modification
TAG_EXIF_IFD = 0x8769
TAG_DATETIME_ORIGINAL = 0x9003     # when the shutter fired
TAG_DATETIME_DIGITIZED = 0x9004
TAG_OFFSET_TIME = 0x9010
TAG_OFFSET_TIME_ORIGINAL = 0x9011
TAG_OFFSET_TIME_DIGITIZED = 0x9012
TAG_SUBSEC_TIME = 0x9290
TAG_SUBSEC_TIME_ORIGINAL = 0x9291
TAG_SUBSEC_TIME_DIGITIZED = 0x9292
TAG_BODY_SERIAL = 0xA431

# Capture-time sources, best first: (datetime tag, subsec tag, offset tag,
# where the datetime tag lives, label stored in the database)
_TIME_SOURCES = (
    (TAG_DATETIME_ORIGINAL, TAG_SUBSEC_TIME_ORIGINAL, TAG_OFFSET_TIME_ORIGINAL, "exif", "exif_original"),
    (TAG_DATETIME_DIGITIZED, TAG_SUBSEC_TIME_DIGITIZED, TAG_OFFSET_TIME_DIGITIZED, "exif", "exif_digitized"),
    (TAG_DATETIME, TAG_SUBSEC_TIME, TAG_OFFSET_TIME, "ifd0", "exif_modified"),
)

# Files created by macOS on memory cards ("._IMG_0001.JPG") are not photos.
_JUNK_PREFIXES = ("._",)


class UnreadableImageError(Exception):
    """The file exists but cannot be read as an image."""


@dataclass
class PhotoInfo:
    file_path: str          # absolute path as found on disk
    path_key: str           # normalised key for uniqueness (see make_path_key)
    file_name: str
    file_size: int
    file_mtime_ns: int
    fingerprint: str
    width: int              # EXIF-oriented, as displayed
    height: int
    exif_orientation: int | None
    capture_time: str | None
    capture_offset: str | None
    capture_time_source: str | None
    camera_make: str | None
    camera_model: str | None
    camera_serial: str | None

    def as_db_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------ paths

def make_path_key(path: str | os.PathLike, pathmod=os.path) -> str:
    """
    Normalised absolute path used to decide whether two paths are the same
    file. On Windows this lower-cases and unifies slashes, because NTFS
    paths are case-insensitive. (pathmod is replaceable for testing.)
    """
    return pathmod.normcase(pathmod.normpath(pathmod.abspath(os.fspath(path))))


def is_under(path_key: str, root_key: str, sep: str = os.sep) -> bool:
    """True if path_key is inside the folder root_key (both from make_path_key)."""
    root = root_key.rstrip(sep)
    return path_key == root or path_key.startswith(root + sep)


# ------------------------------------------------------------ discovery

def _is_hidden(entry: os.DirEntry) -> bool:
    if entry.name.startswith("."):
        return True
    attrs = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)  # Windows only
    return bool(attrs & (stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM))


def iter_photo_files(
    root: str | os.PathLike,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    recursive: bool = True,
    skip_dir_names: tuple[str, ...] = (),
    exclude_dirs: tuple[str | os.PathLike, ...] = (),
    on_error: Callable[[str, Exception], None] | None = None,
) -> Iterator[tuple[str, os.stat_result]]:
    """
    Yield (absolute_path, stat_result) for every supported image under root,
    in a stable, sorted order.

    Skips hidden/system folders and files, macOS "._" files, the folders in
    skip_dir_names, and everything inside exclude_dirs (e.g. the export
    folder, so exported copies are never indexed twice). Symbolic links are
    not followed, and each real folder is visited once, so link loops are
    impossible. Unreadable folders are reported to on_error and skipped.
    """
    exts = tuple(e.lower() for e in extensions)
    skip_names = {n.lower() for n in skip_dir_names}
    exclude_keys = [make_path_key(d) for d in exclude_dirs]
    visited: set[tuple[int, int]] = set()
    stack = [os.path.abspath(os.fspath(root))]

    while stack:
        folder = stack.pop()
        try:
            st = os.stat(folder)
            ident = (st.st_dev, st.st_ino)
            if ident in visited:
                continue
            visited.add(ident)
            with os.scandir(folder) as it:
                entries = sorted(it, key=lambda e: e.name.lower())
        except OSError as exc:
            if on_error:
                on_error(folder, exc)
            continue

        subfolders = []
        for entry in entries:
            try:
                if entry.is_symlink() or _is_hidden(entry):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if recursive and entry.name.lower() not in skip_names:
                        key = make_path_key(entry.path)
                        if not any(is_under(key, ex) for ex in exclude_keys):
                            subfolders.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    name = entry.name.lower()
                    if name.endswith(exts) and not name.startswith(_JUNK_PREFIXES):
                        yield entry.path, entry.stat(follow_symlinks=False)
            except OSError as exc:
                if on_error:
                    on_error(entry.path, exc)
        # Reverse so folders are processed in alphabetical order.
        stack.extend(reversed(subfolders))


# ------------------------------------------------------------ fingerprint

def compute_fingerprint(path: str | os.PathLike, size: int,
                        chunk: int = 64 * 1024) -> str:
    """
    Fast content fingerprint: SHA-256 of the file size plus its first and
    last `chunk` bytes. Identifies the same photo after a move or rename
    without reading whole files. Not a security hash.
    """
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(size - chunk, chunk))
            h.update(f.read(chunk))
    return h.hexdigest()


# ------------------------------------------------------------ EXIF parsing

_DT_RE = re.compile(
    r"^\s*(\d{4})[:\-](\d{2})[:\-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:[.,](\d+))?")
_OFFSET_RE = re.compile(r"^[+-]\d{2}:\d{2}$")


def _clean_text(value) -> str | None:
    """EXIF strings often carry NUL padding or arrive as bytes."""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).replace("\x00", "").strip()
    return text or None


def parse_subsec(value) -> str | None:
    """
    EXIF SubSecTime holds the digits after the decimal point: "45" means
    0.45 s. Returns at least 3 digits (milliseconds), at most 6, so that
    capture times sort correctly as text.
    """
    text = _clean_text(value)
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return digits[:6].ljust(3, "0")


def parse_exif_datetime(value, subsec=None) -> str | None:
    """
    "2026:04:12 08:31:22" + subsec "45"  ->  "2026-04-12T08:31:22.450"
    Returns None for empty, zeroed ("0000:00:00 ...") or invalid dates.
    The value is kept as camera local time; no time-zone conversion.
    """
    text = _clean_text(value)
    if not text:
        return None
    m = _DT_RE.match(text)
    if not m:
        return None
    y, mo, d, h, mi, s, inline_frac = m.groups()
    try:
        dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
    except ValueError:
        return None
    frac = parse_subsec(subsec) or parse_subsec(inline_frac)
    iso = dt.strftime("%Y-%m-%dT%H:%M:%S")
    return f"{iso}.{frac}" if frac else iso


def parse_offset(value) -> str | None:
    text = _clean_text(value)
    return text if text and _OFFSET_RE.match(text) else None


def _orientation(value) -> int | None:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if 1 <= v <= 8 else None


def extract_exif_fields(exif: Image.Exif) -> dict:
    """Pick the fields we store from a Pillow Exif object."""
    try:
        exif_ifd = exif.get_ifd(TAG_EXIF_IFD)
    except Exception:  # noqa: BLE001 - damaged sub-IFD: use what IFD0 has
        exif_ifd = {}

    capture_time = capture_offset = source = None
    for dt_tag, ss_tag, off_tag, where, label in _TIME_SOURCES:
        raw = (exif_ifd if where == "exif" else exif).get(dt_tag)
        parsed = parse_exif_datetime(raw, exif_ifd.get(ss_tag))
        if parsed:
            capture_time, source = parsed, label
            capture_offset = parse_offset(exif_ifd.get(off_tag))
            break

    return {
        "exif_orientation": _orientation(exif.get(TAG_ORIENTATION)),
        "capture_time": capture_time,
        "capture_offset": capture_offset,
        "capture_time_source": source,
        "camera_make": _clean_text(exif.get(TAG_MAKE)),
        "camera_model": _clean_text(exif.get(TAG_MODEL)),
        "camera_serial": _clean_text(exif_ifd.get(TAG_BODY_SERIAL)),
    }


# ------------------------------------------------------------ reading

_OPEN_ERRORS = (UnidentifiedImageError, Image.DecompressionBombError, SyntaxError,
                ValueError, OSError)


def _pil_open(path):
    """
    Open an image with Pillow's own Image.open, so the scanner behaves the
    same whether or not Ultralytics (which replaces Image.open) is loaded.
    See utils/pil_compat.py for the root cause.
    """
    return pillow_open()(path)

def read_photo_info(path: str | os.PathLike, st: os.stat_result | None = None,
                    fingerprint_chunk: int = 64 * 1024) -> PhotoInfo:
    """
    Read size, dimensions and EXIF metadata. Only the file header is
    decoded, not the pixels, so this is fast.
    Raises UnreadableImageError if the file is not a readable image.
    """
    path = os.path.abspath(os.fspath(path))
    st = st or os.stat(path)
    if st.st_size == 0:
        raise UnreadableImageError("File is empty (0 bytes)")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # e.g. "Corrupt EXIF data"
            with _pil_open(path) as img:
                raw_w, raw_h = img.size
                try:
                    fields = extract_exif_fields(img.getexif())
                except Exception:  # noqa: BLE001 - bad EXIF must not lose the photo
                    fields = extract_exif_fields(Image.Exif())
    except _OPEN_ERRORS as exc:
        raise UnreadableImageError(f"{type(exc).__name__}: {exc}") from exc
    except MemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 - any other failure: this file is unreadable,
        raise UnreadableImageError(f"{type(exc).__name__}: {exc}") from exc  # the scan goes on

    # Orientations 5-8 rotate by 90 degrees: displayed width/height swap.
    swap = fields["exif_orientation"] in (5, 6, 7, 8)
    width, height = (raw_h, raw_w) if swap else (raw_w, raw_h)

    return PhotoInfo(
        file_path=path,
        path_key=make_path_key(path),
        file_name=os.path.basename(path),
        file_size=st.st_size,
        file_mtime_ns=st.st_mtime_ns,
        fingerprint=compute_fingerprint(path, st.st_size, fingerprint_chunk),
        width=width,
        height=height,
        **fields,
    )


def load_image_oriented(path: str | os.PathLike) -> Image.Image:
    """
    Fully decode a photo and apply its EXIF orientation. Returns an RGB
    image in the coordinate system used for all stored boxes.
    Raises UnreadableImageError for corrupt or truncated files.
    """
    return load_image_with_orientation(path)[0]


def load_image_with_orientation(path: str | os.PathLike) -> tuple[Image.Image, int | None]:
    """Like load_image_oriented(), also returning the EXIF orientation (1-8 or None)."""
    try:
        with _pil_open(path) as img:
            img.load()                            # decode now, inside the try
            try:
                orientation = _orientation(img.getexif().get(TAG_ORIENTATION))
            except Exception:  # noqa: BLE001 - damaged EXIF: treat as unrotated
                orientation = None
            oriented = ImageOps.exif_transpose(img)
            return oriented.convert("RGB"), orientation
    except _OPEN_ERRORS as exc:
        raise UnreadableImageError(f"{type(exc).__name__}: {exc}") from exc
    except MemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 - see read_photo_info
        raise UnreadableImageError(f"{type(exc).__name__}: {exc}") from exc
