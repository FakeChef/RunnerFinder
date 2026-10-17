"""Small, batch-scoped search helpers for diagnostic ``photo_bibs.csv`` files.

This module deliberately uses CSV files only.  It never changes the diagnostic
CSV or the source photographs; review decisions live in a separate CSV beside
the selected batch.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


_DIGITS = re.compile(r"\d+")
OVERRIDE_FIELDS = ["batch_id", "photo_identity", "bib", "action"]


@dataclass(frozen=True)
class PhotoBibRecord:
    batch_id: str
    photo_no: str
    file_value: str
    bib: str
    numeric_identifier: str
    confidence: str
    readings: str
    region_ids: str
    raw: dict[str, str]


@dataclass(frozen=True)
class ResolvedPhoto:
    path: Path | None
    status: str
    detail: str


@dataclass(frozen=True)
class PhotoSearchGroup:
    """One photo/numeric-bib result with all original OCR records attached."""
    batch_id: str
    photo_identity: str
    resolved: ResolvedPhoto
    numeric_identifier: str
    records: tuple[PhotoBibRecord, ...]


def numeric_part(value: str) -> str:
    """Return the exact digit sequence used for search, preserving zeros."""
    value = (value or "").strip()
    return value if value.isdigit() else "".join(_DIGITS.findall(value)) if len(_DIGITS.findall(value)) == 1 else ""


def resolve_photo_path(file_value: str, csv_path: str | Path,
                       photo_root: str | Path | None = None) -> ResolvedPhoto:
    """Resolve an original path without guessing among duplicate filenames.

    Absolute paths are trusted only when they exist. Relative paths are tried
    relative to the CSV and, if needed, searched below that batch directory.
    Multiple matches are reported as ambiguous.
    """
    raw = (file_value or "").strip()
    csv_path = Path(csv_path).resolve()
    if not raw:
        return ResolvedPhoto(None, "missing", "empty file path")
    candidate = Path(raw)
    if candidate.is_absolute():
        if candidate.is_file():
            return ResolvedPhoto(candidate.resolve(), "unique", "original full path")
        return ResolvedPhoto(None, "missing", f"original path does not exist: {candidate}")

    roots = [csv_path.parent]
    if photo_root:
        root = Path(photo_root).expanduser().resolve()
        if root.is_dir() and root not in roots:
            roots.append(root)
    direct_matches = [(root / candidate).resolve() for root in roots
                      if (root / candidate).is_file()]
    if len(set(direct_matches)) == 1:
        return ResolvedPhoto(direct_matches[0], "unique", "relative path resolved from batch/photo root")
    matches = [p.resolve() for root in roots for p in root.rglob(candidate.name) if p.is_file()]
    matches = sorted(set(matches), key=lambda p: str(p).lower())
    if len(matches) == 1:
        return ResolvedPhoto(matches[0], "unique", "one relative filename match")
    if len(matches) > 1:
        return ResolvedPhoto(None, "ambiguous", "duplicate filenames: " + "; ".join(map(str, matches)))
    return ResolvedPhoto(None, "missing", f"no file found for relative path: {raw}")


def load_records(csv_path: str | Path) -> list[PhotoBibRecord]:
    csv_path = Path(csv_path).resolve()
    batch_id = str(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = csv.DictReader(fh)
        out = []
        for row in rows:
            bib = (row.get("bib") or "").strip()
            numeric = (row.get("numeric_identifier") or "").strip() or numeric_part(bib)
            out.append(PhotoBibRecord(batch_id, (row.get("photo_no") or "").strip(),
                                      (row.get("file") or row.get("file_name") or "").strip(),
                                      bib, numeric, (row.get("best_confidence") or "").strip(),
                                      (row.get("readings") or "").strip(),
                                      (row.get("region_ids") or "").strip(), dict(row)))
    return out


def search_records(records: list[PhotoBibRecord], query: str) -> list[PhotoBibRecord]:
    """Exact numeric matching; ``123`` does not match ``00123``."""
    q = (query or "").strip()
    if not q.isdigit():
        return []
    return [r for r in records if r.numeric_identifier == q]


def filter_records(records: list[PhotoBibRecord], min_length: int | None = None,
                   max_length: int | None = None,
                   excluded_numeric: set[str] | None = None) -> tuple[list[PhotoBibRecord], list[PhotoBibRecord]]:
    """Return (kept, filtered) without changing source records."""
    excluded_numeric = excluded_numeric or set()
    kept, filtered = [], []
    for record in records:
        n = record.numeric_identifier
        reject = ((min_length is not None and len(n) < min_length)
                  or (max_length is not None and len(n) > max_length)
                  or n in excluded_numeric)
        (filtered if reject else kept).append(record)
    return kept, filtered


def group_records(records: list[PhotoBibRecord], csv_path: str | Path,
                  photo_root: str | Path | None = None) -> list[PhotoSearchGroup]:
    """Deduplicate by batch, resolved photo identity and exact numeric bib."""
    csv_path = Path(csv_path).resolve()
    groups: dict[tuple[str, str, str], list[PhotoBibRecord]] = {}
    resolved_by_key: dict[tuple[str, str, str], ResolvedPhoto] = {}
    for record in records:
        resolved = resolve_photo_path(record.file_value, csv_path, photo_root)
        identity = str(resolved.path) if resolved.path else f"{resolved.status}:{record.file_value}"
        key = (record.batch_id, identity, record.numeric_identifier)
        groups.setdefault(key, []).append(record)
        resolved_by_key[key] = resolved
    return [PhotoSearchGroup(k[0], k[1], resolved_by_key[k], k[2], tuple(v))
            for k, v in sorted(groups.items(), key=lambda item: item[0])]


def override_path(csv_path: str | Path) -> Path:
    return Path(csv_path).resolve().parent / "review_overrides.csv"


def load_overrides(csv_path: str | Path) -> dict[tuple[str, str, str], str]:
    path = override_path(csv_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {(r.get("batch_id", ""), r.get("photo_identity", ""), r.get("bib", "")): r.get("action", "")
                for r in csv.DictReader(fh) if r.get("action") in {"reject", "restore"}}


def save_override(csv_path: str | Path, photo_identity: str, bib: str, action: str) -> Path:
    if action not in {"reject", "restore"}:
        raise ValueError("action must be reject or restore")
    path = override_path(csv_path)
    existing = load_overrides(csv_path)
    key = (str(Path(csv_path).resolve()), photo_identity, bib)
    existing[key] = action
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OVERRIDE_FIELDS)
        writer.writeheader()
        for (batch_id, photo, candidate), value in sorted(existing.items()):
            writer.writerow({"batch_id": batch_id, "photo_identity": photo, "bib": candidate, "action": value})
    return path


def is_rejected(overrides: dict[tuple[str, str, str], str], csv_path: str | Path,
                photo_identity: str, bib: str) -> bool:
    return overrides.get((str(Path(csv_path).resolve()), photo_identity, bib)) == "reject"


def open_in_explorer(path: str | Path) -> None:
    """Open and select one existing original photo in Windows Explorer."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    subprocess.Popen(["explorer.exe", f"/select,{p}"])  # noqa: S603 - fixed executable, one user path
