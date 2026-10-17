"""
Export matching photos into a photographer-friendly folder layout:

    {export_root}/{Event_Name}/Bib_{Number}/photo.jpg

Export only ever COPIES; originals are never moved or modified.
Path building and collision-free naming are implemented now; the copy
step is implemented in Phase 7.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')     # not allowed on Windows
_RESERVED = {"CON", "PRN", "AUX", "NUL",
             *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}


def safe_folder_name(name: str, fallback: str = "Unnamed") -> str:
    """Make any text safe to use as a Windows folder name."""
    cleaned = _INVALID_CHARS.sub("_", name or "").strip().rstrip(". ")
    cleaned = re.sub(r"\s+", "_", cleaned)
    if not cleaned:
        cleaned = fallback
    if cleaned.split(".")[0].upper() in _RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned[:100]


def bib_export_folder(export_root: Path, event_name: str, bib_number: str) -> Path:
    """Return export_root / Event_Name / Bib_<number>. Leading zeros are kept."""
    return (Path(export_root)
            / safe_folder_name(event_name, "Event")
            / f"Bib_{safe_folder_name(str(bib_number), 'Unknown')}")


def unique_destination(dest_dir: Path, file_name: str, taken: set[str]) -> Path:
    """
    Choose a file name in dest_dir that is not used yet. Two photos called
    IMG_0001.JPG from different cards become IMG_0001.JPG and IMG_0001_2.JPG.
    `taken` holds lower-case names already planned in this export (Windows
    names are case-insensitive) and is updated.
    """
    stem, ext = os.path.splitext(file_name)
    candidate, n = file_name, 1
    while candidate.lower() in taken or (Path(dest_dir) / candidate).exists():
        n += 1
        candidate = f"{stem}_{n}{ext}"
    taken.add(candidate.lower())
    return Path(dest_dir) / candidate


def export_photos(photo_paths: list[Path], export_root: Path,
                  event_name: str, bib_number: str) -> list[Path]:
    """Copy photos into the bib folder. Implemented in Phase 7."""
    raise NotImplementedError("Phase 7")
