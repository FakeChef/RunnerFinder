"""
Detection diagnostic run (Phase 4): process a sample of real photos,
write debug images and a report.

This measures what the detection stage DOES (people found, regions
produced, time per image). It does not measure accuracy: there is no
ground truth here, and bib recognition (OCR) is not implemented yet.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from config import AppConfig
from detection.bib_detector import RegionFinder, crop_region
from detection.debug_viz import ensure_inside, write_debug
from detection.person_detector import PersonDetector
from export.exporter import safe_folder_name
from scanner.photo_scanner import (UnreadableImageError, iter_photo_files,
                                   load_image_with_orientation, make_path_key)
from utils.sysinfo import ai_summary, system_summary


@dataclass
class ImageRow:
    order: int
    file: str
    status: str                      # "ok" | "failed"
    error: str = ""
    width: int = 0
    height: int = 0
    orientation: int | None = None
    persons_detected: int = 0
    persons_kept: int = 0
    persons_skipped_small: int = 0
    max_person_confidence: float | None = None
    regions_person_body: int = 0
    regions_full_image: int = 0
    crops_touching_edge: int = 0
    fallback: str = ""
    decode_ms: float = 0.0
    resize_ms: float = 0.0
    detect_ms: float = 0.0
    regions_ms: float = 0.0
    crop_ms: float = 0.0
    total_ms: float = 0.0            # decode + resize + detect + regions + crop
    debug_write_ms: float = 0.0      # NOT included in total_ms
    overview_image: str = ""


@dataclass
class BenchmarkReport:
    folder: str
    run_dir: str
    started: str
    settings: dict
    system: dict
    model_load_s: float | None
    images_found: int = 0
    rows: list[ImageRow] = field(default_factory=list)


def select_files(files: list, limit: int | None, spread: bool = True) -> list:
    """First `limit` files, or `limit` files spread evenly over the whole list."""
    if not limit or len(files) <= limit:
        return list(files)
    if not spread or limit == 1:
        return list(files[:limit])
    n = len(files)
    idx = sorted({round(i * (n - 1) / (limit - 1)) for i in range(limit)})
    return [files[i] for i in idx]


def _debug_name(order: int, path: str) -> str:
    stem = safe_folder_name(Path(path).stem, "photo")[:40]
    digest = hashlib.sha1(make_path_key(path).encode("utf-8")).hexdigest()[:6]
    return f"{order:03d}_{stem}_{digest}"


def run_detection_benchmark(folder: str | os.PathLike, cfg: AppConfig, detector: PersonDetector,
                            limit: int | None = 50, spread: bool = True, debug: bool = True,
                            model_load_s: float | None = None,
                            progress: Callable[[int, int, str], None] | None = None) -> BenchmarkReport:
    folder = os.path.abspath(os.fspath(folder))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_inside(cfg.paths.debug_dir / f"detect_{stamp}", cfg.paths.data_dir)
    if make_path_key(run_dir).startswith(make_path_key(folder) + os.sep):
        raise ValueError("The data folder is inside the photo folder; choose another photo folder.")
    run_dir.mkdir(parents=True, exist_ok=True)

    d = cfg.detection
    report = BenchmarkReport(
        folder=folder, run_dir=str(run_dir), started=datetime.now().isoformat(timespec="seconds"),
        settings={"detector": getattr(detector, "name", type(detector).__name__),
                  "detection_image_size": d.detection_image_size,
                  "person_conf_threshold": d.person_conf_threshold,
                  "min_person_size_px": d.min_person_size_px,
                  "crop_strategy": d.crop_strategy, "crop_fractions": d.crop_fractions(),
                  "extend_crop_at_image_edges": d.extend_crop_at_image_edges,
                  "full_image_fallback": d.full_image_fallback,
                  "full_image_tile_size": d.full_image_tile_size,
                  "limit": limit, "selection": "spread" if spread else "first", "debug": debug},
        system={**system_summary(), **ai_summary()}, model_load_s=model_load_s)

    all_files = [p for p, _ in iter_photo_files(folder, cfg.scan.extensions, cfg.scan.recursive,
                                                cfg.scan.skip_dir_names,
                                                (cfg.paths.default_export_root, cfg.paths.data_dir))]
    report.images_found = len(all_files)
    selected = select_files(all_files, limit, spread)
    finder = RegionFinder(d, detector)

    for order, path in enumerate(selected, start=1):
        if progress:
            progress(order, len(selected), path)
        row = ImageRow(order=order, file=path, status="ok")
        try:
            t = time.perf_counter()
            image, orientation = load_image_with_orientation(path)
            row.decode_ms = (time.perf_counter() - t) * 1000
            result = finder.find(image)
            t = time.perf_counter()
            for region in result.regions:      # what Phase 5 will hand to OCR
                crop_region(image, region)
            row.crop_ms = (time.perf_counter() - t) * 1000
        except UnreadableImageError as exc:
            row.status, row.error = "failed", str(exc)
            report.rows.append(row)
            continue
        except Exception as exc:  # noqa: BLE001 - one bad photo must not stop the run
            row.status, row.error = "failed", f"{type(exc).__name__}: {exc}"
            report.rows.append(row)
            continue

        row.width, row.height = result.image_size
        row.orientation = orientation
        row.resize_ms = result.timings_ms["resize_ms"]
        row.detect_ms = result.timings_ms["detect_ms"]
        row.regions_ms = result.timings_ms["regions_ms"]
        row.total_ms = row.decode_ms + row.resize_ms + row.detect_ms + row.regions_ms + row.crop_ms
        row.persons_detected = len(result.persons)
        row.persons_kept = len(result.kept_persons)
        row.persons_skipped_small = row.persons_detected - row.persons_kept
        row.max_person_confidence = round(max((p.confidence for p in result.persons), default=0), 4) \
            if result.persons else None
        row.regions_person_body = sum(r.source == "person_body" for r in result.regions)
        row.regions_full_image = sum(r.source == "full_image" for r in result.regions)
        row.crops_touching_edge = sum(any("edge" in n for n in r.notes)
                                      for r in result.regions if r.source == "person_body")
        row.fallback = (result.fallback_reason or
                        ("pending (after OCR)" if result.full_image_fallback_pending else ""))
        if debug:
            t = time.perf_counter()
            timings = {"decode_ms": row.decode_ms, **result.timings_ms,
                       "crop_ms": row.crop_ms, "total_ms": row.total_ms}
            try:
                paths = write_debug(
                    run_dir / "images", cfg.paths.data_dir, _debug_name(order, path), path,
                    image, result, timings, orientation, cfg.debug.overlay_max_side,
                    cfg.debug.overlay_jpeg_quality, cfg.debug.crop_jpeg_quality,
                    time_note=" (first image: includes model warm-up)" if order == 1 else "")
                row.overview_image = paths["overview"]
            except Exception as exc:  # noqa: BLE001 - debug output must not stop the run
                row.error = f"debug images not written: {type(exc).__name__}: {exc}"
            row.debug_write_ms = (time.perf_counter() - t) * 1000
        report.rows.append(row)

    write_report_files(report)
    return report


# ------------------------------------------------------------------ report

def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    p90 = v[min(len(v) - 1, round(0.9 * (len(v) - 1)))]
    return {"n": len(v), "mean": statistics.fmean(v), "median": statistics.median(v),
            "p90": p90, "min": v[0], "max": v[-1]}


def summarize(report: BenchmarkReport) -> dict:
    ok = [r for r in report.rows if r.status == "ok"]
    timed = ok[1:] if len(ok) > 1 else ok          # first image includes warm-up
    persons_hist = {"0": 0, "1": 0, "2": 0, "3-5": 0, "6+": 0}
    for r in ok:
        k = r.persons_kept
        persons_hist["0" if k == 0 else "1" if k == 1 else "2" if k == 2 else "3-5" if k <= 5 else "6+"] += 1
    return {
        "images_found_in_folder": report.images_found,
        "images_selected": len(report.rows),
        "images_processed": len(ok),
        "images_failed": len(report.rows) - len(ok),
        "images_with_person_detected": sum(r.persons_detected > 0 for r in ok),
        "images_with_person_kept": sum(r.persons_kept > 0 for r in ok),
        "images_without_person": sum(r.persons_kept == 0 for r in ok),
        "person_detections_total": sum(r.persons_detected for r in ok),
        "person_detections_kept": sum(r.persons_kept for r in ok),
        "person_detections_skipped_small": sum(r.persons_skipped_small for r in ok),
        "kept_persons_per_image": persons_hist,
        "ocr_regions_person_body": sum(r.regions_person_body for r in ok),
        "ocr_regions_full_image_tiles": sum(r.regions_full_image for r in ok),
        "images_using_full_image_fallback": sum(r.regions_full_image > 0 for r in ok),
        "images_with_fallback_pending_after_ocr": sum(r.fallback.startswith("pending") for r in ok),
        "crops_at_image_edge": sum(r.crops_touching_edge for r in ok),
        "timing_note": ("per-image times exclude the first processed image (warm-up) and "
                        "exclude model loading and debug-image writing"),
        "time_total_ms": _stats([r.total_ms for r in timed]),
        "time_decode_ms": _stats([r.decode_ms for r in timed]),
        "time_resize_ms": _stats([r.resize_ms for r in timed]),
        "time_detect_ms": _stats([r.detect_ms for r in timed]),
        "time_crop_ms": _stats([r.crop_ms for r in timed]),
        "first_image_total_ms": ok[0].total_ms if ok else None,
        "debug_write_ms": _stats([r.debug_write_ms for r in ok if r.debug_write_ms]),
    }


def _fmt(st: dict) -> str:
    return "n/a" if not st else (f"mean {st['mean']:.0f} | median {st['median']:.0f} | "
                                 f"p90 {st['p90']:.0f} | min {st['min']:.0f} | max {st['max']:.0f} "
                                 f"(n={st['n']})")


def report_text(report: BenchmarkReport) -> str:
    s = summarize(report)
    sysi = report.system
    lines = [
        "# Runner Photo Finder - detection diagnostic report (Phase 4)",
        "",
        "This report describes what the person-detection stage did on these photos.",
        "It does NOT measure bib recognition accuracy (OCR is not implemented yet),",
        "and person counts are not an accuracy figure: there is no ground truth here.",
        "",
        f"- Started: {report.started}",
        f"- Photo folder: `{report.folder}`",
        f"- Debug output: `{report.run_dir}`",
        "",
        "## System",
        f"- OS: {sysi.get('os')}",
        f"- CPU: {sysi.get('cpu')} ({sysi.get('logical_cores')} logical cores), RAM: {sysi.get('ram_gb')} GB",
        f"- Python {sysi.get('python')}, torch {sysi.get('torch', 'n/a')} "
        f"(threads {sysi.get('torch_threads', 'n/a')}), ultralytics {sysi.get('ultralytics', 'n/a')}, "
        f"CUDA GPU: {sysi.get('cuda_gpu') or 'none'}",
        "",
        "## Settings",
        *[f"- {k}: {v}" for k, v in report.settings.items()],
        "",
        "## Results",
        f"- Images found in folder: {s['images_found_in_folder']}",
        f"- Images selected: {s['images_selected']}",
        f"- Images processed: {s['images_processed']}",
        f"- Failed / unreadable images: {s['images_failed']}",
        f"- Images with at least one person detected: {s['images_with_person_detected']}",
        f"- Images with at least one person kept (large enough): {s['images_with_person_kept']}",
        f"- Total person detections: {s['person_detections_total']} "
        f"(kept {s['person_detections_kept']}, skipped as too small {s['person_detections_skipped_small']})",
        f"- Kept persons per image: {s['kept_persons_per_image']}",
        f"- OCR regions produced: {s['ocr_regions_person_body']} body crops, "
        f"{s['ocr_regions_full_image_tiles']} full-image tiles",
        f"- Images using full-image fallback now: {s['images_using_full_image_fallback']}; "
        f"with fallback pending after OCR: {s['images_with_fallback_pending_after_ocr']}",
        f"- Body crops touching an image edge (partly visible runners): {s['crops_at_image_edge']}",
        "",
        "## Processing time per image (milliseconds)",
        f"Method: wall-clock (time.perf_counter) per stage, one pass, {s['timing_note']}.",
        f"Model loading: {report.model_load_s:.2f} s" if report.model_load_s is not None
        else "Model loading: not measured",
        f"- Total (decode + resize + detect + crops): {_fmt(s['time_total_ms'])}",
        f"- Decode JPEG/PNG + EXIF rotation: {_fmt(s['time_decode_ms'])}",
        f"- Downscale for detection: {_fmt(s['time_resize_ms'])}",
        f"- Person detection (YOLO): {_fmt(s['time_detect_ms'])}",
        f"- Cutting crops: {_fmt(s['time_crop_ms'])}",
        f"- First image (incl. warm-up): " +
        (f"{s['first_image_total_ms']:.0f}" if s['first_image_total_ms'] is not None else "n/a"),
        f"- Writing debug images (not included above): {_fmt(s['debug_write_ms'])}",
        "",
        "## Images",
        "| # | File | Status | Size | Persons kept/detected | Regions | Total ms | Overview image |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report.rows:
        if r.status == "ok":
            regions = f"{r.regions_person_body} body" + (f" + {r.regions_full_image} tiles"
                                                         if r.regions_full_image else "")
            lines.append(f"| {r.order} | {Path(r.file).name} | ok | {r.width}x{r.height} | "
                         f"{r.persons_kept}/{r.persons_detected} | {regions} | {r.total_ms:.0f} | "
                         f"{Path(r.overview_image).name if r.overview_image else ''} |")
        else:
            lines.append(f"| {r.order} | {Path(r.file).name} | FAILED: {r.error} | | | | | |")
    return "\n".join(lines) + "\n"


def write_report_files(report: BenchmarkReport) -> None:
    run_dir = Path(report.run_dir)
    (run_dir / "report.md").write_text(report_text(report), encoding="utf-8")
    with open(run_dir / "images.csv", "w", newline="", encoding="utf-8-sig") as f:   # BOM: Excel
        writer = csv.DictWriter(f, fieldnames=list(asdict(report.rows[0]).keys()) if report.rows
                                else ["order"])
        writer.writeheader()
        for r in report.rows:
            writer.writerow({k: (round(v, 1) if isinstance(v, float) else v)
                             for k, v in asdict(r).items()})
    (run_dir / "report.json").write_text(json.dumps(
        {"summary": summarize(report), **{k: v for k, v in asdict(report).items() if k != "rows"},
         "images": [asdict(r) for r in report.rows]}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
