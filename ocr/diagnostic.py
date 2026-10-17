"""
OCR diagnostic run (Phase 5): detection + OCR on a sample of real photos.

Writes, under data/debug/ocr_<date>_<time>/ :
  report.md / report.json      summary, settings, system, timing method
  photos.csv                   one row per photo (deduplicated bibs, status, timings)
  photo_bibs.csv               one row per (photo, bib), with empty review columns
  readings.csv                 every OCR candidate: raw text, normalised value,
                               confidence, coordinates, source crop, accepted /
                               rejection reason
  regions.csv                  every crop/tile sent to OCR: status, lines, time
  ground_truth_review.xlsx     one row per photo for manual correction
  ground_truth_review.csv      same content as CSV
  REVIEW_INSTRUCTIONS.txt      how to fill in the review sheet
  images/                      overview images with OCR results, and crops

It does NOT measure OCR accuracy or search recall: that needs the
ground truth produced by the review sheet.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from config import AppConfig
from detection.benchmark import _debug_name, select_files
from detection.bib_detector import RegionFinder
from detection.debug_viz import BIB_OK, BIB_REJECTED, TILE, ensure_inside, write_debug
from detection.person_detector import PersonDetector
from ocr.bib_reader import OCREngine
from ocr.candidates import PART_OF_JOINED
from ocr.photo_ocr import PhotoOCR, PhotoOCRResult
from scanner.photo_scanner import (UnreadableImageError, iter_photo_files,
                                   load_image_with_orientation, make_path_key)
from utils.sysinfo import ai_summary, system_summary
from utils.xlsx_writer import write_xlsx

REVIEW_HEADERS = ["photo_no", "file_name", "relative_path", "ai_bibs", "ai_confidences",
                  "ai_status", "true_bibs", "visible_but_unreadable", "tags", "notes",
                  "overview_image"]

REVIEW_INSTRUCTIONS = """How to fill in ground_truth_review.xlsx
========================================

Open ground_truth_review.xlsx (not the .csv: Excel removes leading zeros
from CSV files, e.g. 00123 becomes 123). All columns are formatted as text.

One row per photo. Look at the photo (or the overview image), then fill in:

  true_bibs               every bib number YOU can read in the photo,
                          separated by ";" (e.g. 10258;10311;UT001).
                          Copy correct AI values, fix wrong ones, add missed
                          ones. Leave empty if no bib is readable.
  visible_but_unreadable  how many bibs are visible but too small, blurred
                          or covered to read with certainty (a number).
  tags                    optional, separated by ";": front, side, back,
                          group, blur, trail, waist, thigh, occluded, ...
  notes                   anything else.

Do not change photo_no, file_name or relative_path.
ai_bibs is what the software read; it is there to save typing, not as the
answer. Please check every row, including rows where ai_bibs is empty.

Save as .xlsx. This file becomes the ground truth for measuring OCR
accuracy and search recall in a later step; until then no accuracy figures
are reported.
"""


@dataclass
class PhotoRow:
    order: int
    file: str
    status: str
    error: str = ""
    result: PhotoOCRResult | None = None
    orientation: int | None = None
    decode_ms: float = 0.0
    detect_ms: float = 0.0
    crop_ocr_ms: float = 0.0
    fallback_ocr_ms: float = 0.0
    total_ms: float = 0.0
    debug_write_ms: float = 0.0
    overview_image: str = ""


@dataclass
class OCRDiagnostic:
    folder: str
    run_dir: str
    started: str
    settings: dict
    system: dict
    model_load_s: dict
    images_found: int = 0
    rows: list[PhotoRow] = field(default_factory=list)
    pipeline_used: str = ""          # V1.3: s3 | prefetch | sequential (what actually ran)
    pipeline_fallback: str = ""      # V1.3: why S3 was not used although requested


def _marks(result: PhotoOCRResult) -> list:
    """Overview marks: one per OCR line that contains a digit or was accepted."""
    marks, seen = [], {}
    for r in result.readings:
        key = (r.region_id, r.line_index)
        if r.accepted:
            seen.setdefault(key, [r.line_box, [], None])[1].append(f"{r.bib} {r.ocr_confidence:.2f}")
        elif r.reason != PART_OF_JOINED and any(ch.isdigit() for ch in r.normalized):
            seen.setdefault(key, [r.line_box, [], None])
            if seen[key][2] is None:
                seen[key][2] = f"x {r.normalized} ({r.reason})"
    for box, accepted, rejected in seen.values():
        if accepted:
            marks.append((box, " ".join(dict.fromkeys(accepted)), BIB_OK))
        elif rejected:
            marks.append((box, rejected, BIB_REJECTED))
    det_ids = {id(r) for r in result.detection.regions}
    for reg in result.regions:                         # tiles added by the fallback
        if id(reg.region) not in det_ids:
            marks.append((reg.region.box, f"fallback {reg.region_id}", TILE))
    return marks


def run_settings(cfg: AppConfig, detector, engine, registered=None, limit=None,
                 spread: bool = True, debug: bool = True) -> dict:
    """The settings recorded for a run (report.json, analysis_runs.settings_json).
    database/results_store.settings_hash() picks the result-affecting ones."""
    d, o = cfg.detection, cfg.ocr
    return {"processing_mode": d.processing_mode,
            "detector": getattr(detector, "name", type(detector).__name__),
            "ocr_engine": getattr(engine, "name", type(engine).__name__),
            "ocr_device_requested": o.device,
            "ocr_device_info": getattr(engine, "device_info", None) or None,
            "ocr_device_note": getattr(engine, "device_note", None),
            "crop_strategy": d.crop_strategy, "crop_fractions": d.crop_fractions(),
            "person_conf_threshold": d.person_conf_threshold,
            "min_person_size_px": d.min_person_size_px,
            "full_image_fallback": d.full_image_fallback,
            "full_image_min_ocr_confidence": d.full_image_min_ocr_confidence,
            "full_image_require_person_overlap": d.full_image_require_person_overlap,
            "min_ocr_confidence": o.min_ocr_confidence, "enable_mkldnn": o.enable_mkldnn,
            "text_det_limit_side_len": o.text_det_limit_side_len,
            "text_det_limit_type": o.text_det_limit_type,
            "fix_digit_confusions": o.fix_digit_confusions,
            "bib_length": (cfg.bib.min_length, cfg.bib.max_length),
            "bib_max_letters": cfg.bib.max_letters, "bib_patterns": cfg.bib.patterns,
            "registered_bibs": len(registered) if registered is not None else None,
            "center_crop": (d.center_crop_width, d.center_crop_height),
            "ranking_weights": (d.rank_area_weight, d.rank_center_weight,
                                d.rank_confidence_weight),
            "paddle_cache": os.environ.get("PADDLE_PDX_CACHE_HOME"),
            "limit": limit, "selection": "spread" if spread else "first", "debug": debug}


def run_ocr_diagnostic(folder, cfg: AppConfig, detector: PersonDetector, engine: OCREngine,
                       limit: int | None = 50, spread: bool = True, debug: bool = True,
                       registered: set[str] | None = None, model_load_s: dict | None = None,
                       progress: Callable[[int, int, str], None] | None = None,
                       store=None, files: list[str] | None = None,
                       on_decoded: Callable[[str, object], None] | None = None,
                       prefetch: int = 0, three_stage: bool = False,
                       pipeline_info: dict | None = None) -> OCRDiagnostic:
    """
    files (optional): process exactly these photos (used by "index_cli.py
    analyze"); limit/spread are then ignored. Default: sample `folder`.
    on_decoded (optional): called as on_decoded(path, image) right after each
    photo is decoded and EXIF-rotated, before OCR (used for thumbnails, so a
    photo is never decoded twice). Its errors are ignored.
    store (optional, e.g. database.results_store.OCRRunStore): store.begin(diag,
    selected_paths) is called before the first photo and may refuse the run;
    store.save(row) after EVERY photo, so finished photos are kept if the run
    stops; store.end(completed) at the end. CSV outputs are the same either way.
    prefetch (V1.2): decode this many photos ahead in a producer thread (0 = off,
    V1.1 behaviour). Not part of the run settings: it does not change results.
    three_stage (V1.3, with prefetch > 0): person detection also runs ahead, in its own
    thread (scanner/pipeline3.py). Not part of the run settings either.
    """
    folder = os.path.abspath(os.fspath(folder))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_inside(cfg.paths.debug_dir / f"ocr_{stamp}", cfg.paths.data_dir)
    if make_path_key(run_dir).startswith(make_path_key(folder) + os.sep):
        raise ValueError("The data folder is inside the photo folder; choose another photo folder.")
    run_dir.mkdir(parents=True, exist_ok=True)

    d, o = cfg.detection, cfg.ocr
    diag = OCRDiagnostic(
        folder=folder, run_dir=str(run_dir), started=datetime.now().isoformat(timespec="seconds"),
        settings=run_settings(cfg, detector, engine, registered, limit, spread, debug),
        system={**system_summary(), **ai_summary()}, model_load_s=model_load_s or {})

    if files is None:
        files = [p for p, _ in iter_photo_files(folder, cfg.scan.extensions, cfg.scan.recursive,
                                                cfg.scan.skip_dir_names,
                                                (cfg.paths.default_export_root, cfg.paths.data_dir))]
        selected = select_files(files, limit, spread)
    else:
        selected = list(files)
    diag.images_found = len(files)
    pipeline = PhotoOCR(cfg, RegionFinder(d, detector, d.processing_mode), engine, registered)

    if store is not None:
        store.begin(diag, selected)
    completed = False
    try:
        _run_photos(diag, cfg, pipeline, selected, run_dir, debug, progress, store, on_decoded, prefetch,
                    three_stage, pipeline_info)
        completed = True
    finally:
        if store is not None:
            store.end(completed)

    write_outputs(diag)
    return diag


def _run_photos(diag, cfg, pipeline, selected, run_dir, debug, progress, store,
                on_decoded=None, prefetch: int = 0, three_stage: bool = False,
                pipeline_info: dict | None = None) -> None:
    """prefetch > 0 (V1.2): a producer thread decodes up to `prefetch` photos ahead
    (scanner/prefetch.py) - same functions, same order, same results. 0: V1.1 behaviour.
    three_stage (V1.3, needs prefetch > 0): a second thread also runs the person detection ahead
    (scanner/pipeline3.py); this thread then only runs OCR and commits."""
    def note(used: str, fallback: str = "") -> None:      # V1.3: report what actually runs
        diag.pipeline_used, diag.pipeline_fallback = used, fallback
        if pipeline_info is not None:
            pipeline_info.update(pipeline_used=used, pipeline_fallback=fallback)

    if prefetch <= 0:
        note("sequential")
        _photo_loop(diag, cfg, pipeline, selected, run_dir, debug, progress, store, on_decoded, None)
        return
    fallback = ""
    if three_stage:
        from scanner.pipeline3 import DetectionPipeline, openmp_single_thread_hook, s3_preconditions
        hook, reason = openmp_single_thread_hook()
        if "torch" in sys.modules:
            # V1.3 integration fail-safe: S3 only with the validated OpenMP configuration; otherwise the
            # V1.2 prefetch pipeline below (identical results), with the reason reported.
            ok, reason = s3_preconditions()
            if not ok:
                fallback, three_stage = reason, False
        if three_stage:
            note("s3")
            with DetectionPipeline(selected, pipeline.finder.find, on_decoded, depth=prefetch,
                                   before_detect=hook) as photos:   # closed on cancel / error
                _photo_loop(diag, cfg, pipeline, selected, run_dir, debug, progress, store, on_decoded, photos)
            return
    note("prefetch", fallback)
    from scanner.prefetch import PhotoPrefetcher
    with PhotoPrefetcher(selected, on_decoded, depth=prefetch) as photos:   # closed on cancel / error too
        _photo_loop(diag, cfg, pipeline, selected, run_dir, debug, progress, store, on_decoded, photos)


def _photo_loop(diag, cfg, pipeline, selected, run_dir, debug, progress, store,
                on_decoded, photos) -> None:
    for order, path in enumerate(selected, start=1):
        if progress:
            progress(order, len(selected), path)
        row = PhotoRow(order, path, "ok")
        try:
            t = time.perf_counter()
            if photos is None:
                image, row.orientation = load_image_with_orientation(path)
                row.decode_ms = (time.perf_counter() - t) * 1000
                if on_decoded is not None:
                    try:
                        on_decoded(path, image)
                    except Exception:  # noqa: BLE001 - e.g. a thumbnail problem must not fail the photo
                        pass
            else:                     # decoded (and thumbnail made) by the prefetch thread
                taken = photos.take(path)
                image, row.orientation, row.decode_ms = taken[:3]
            if photos is not None and len(taken) > 3:   # three-stage: detected by the detection thread
                res = pipeline.process(image, detection=taken[3])
            else:
                res = pipeline.process(image)
        except UnreadableImageError as exc:
            row.status, row.error = "failed", str(exc)
            diag.rows.append(row)
            if store is not None:
                store.save(row)
            continue
        except Exception as exc:  # noqa: BLE001 - one bad photo must not stop the run
            row.status, row.error = "failed", f"{type(exc).__name__}: {exc}"
            diag.rows.append(row)
            if store is not None:
                store.save(row)
            continue
        row.result = res
        tm = res.timings_ms
        row.detect_ms = tm.get("resize_ms", 0) + tm.get("detect_ms", 0) + tm.get("regions_ms", 0)
        row.crop_ocr_ms = tm.get("ocr_ms", 0)
        row.fallback_ocr_ms = tm.get("fallback_ocr_ms", 0)
        row.total_ms = row.decode_ms + row.detect_ms + row.crop_ocr_ms + row.fallback_ocr_ms
        if debug:
            t = time.perf_counter()
            bibs = ", ".join(f"{b.bib} ({b.best_confidence:.2f}, x{b.readings})" for b in res.bibs)
            header = [f"OCR: {bibs or 'no valid bib read'} | fallback: {res.fallback_used or 'not used'} "
                      f"| OCR {row.crop_ocr_ms + row.fallback_ocr_ms:.0f} ms | total {row.total_ms:.0f} ms"]
            try:
                paths = write_debug(run_dir / "images", cfg.paths.data_dir, _debug_name(order, path),
                                    path, image, res.detection,
                                    {"decode_ms": row.decode_ms, **res.detection.timings_ms,
                                     "total_ms": row.total_ms},
                                    row.orientation, cfg.debug.overlay_max_side,
                                    cfg.debug.overlay_jpeg_quality, cfg.debug.crop_jpeg_quality,
                                    time_note=" (first photo: includes warm-up)" if order == 1 else "",
                                    marks=_marks(res), extra_header=header,
                                    crop_regions=[r.region for r in res.regions])
                row.overview_image = paths["overview"]
            except Exception as exc:  # noqa: BLE001
                row.error = f"debug images not written: {type(exc).__name__}: {exc}"
            row.debug_write_ms = (time.perf_counter() - t) * 1000
        diag.rows.append(row)
        if store is not None:
            store.save(row)


# ------------------------------------------------------------------ outputs

def _xywh(box) -> list[int]:
    return list(box.xywh())


def _stats(values):
    if not values:
        return {}
    v = sorted(values)
    return {"n": len(v), "mean": statistics.fmean(v), "median": statistics.median(v),
            "p90": v[min(len(v) - 1, round(0.9 * (len(v) - 1)))], "min": v[0], "max": v[-1]}


def summarize(diag: OCRDiagnostic) -> dict:
    ok = [r for r in diag.rows if r.status == "ok"]
    timed = ok[1:] if len(ok) > 1 else ok
    regions = [g for r in ok for g in r.result.regions]
    readings = [x for r in ok for x in r.result.readings]
    accepted = [x for x in readings if x.accepted]
    pairs = sum(len(r.result.bibs) for r in ok)
    reasons, region_status, crop_ms = {}, {}, {"person_body": [], "center_crop": [], "full_image": []}
    for x in readings:
        if not x.accepted:
            reasons[x.reason] = reasons.get(x.reason, 0) + 1
    for g in regions:
        key = f"{g.region.source}: {g.status}"
        region_status[key] = region_status.get(key, 0) + 1
        if g.status not in ("crop_too_small_for_ocr",):
            crop_ms[g.region.source].append(g.ocr_ms)
    return {
        "photos_found_in_folder": diag.images_found,
        "photos_selected": len(diag.rows),
        "photos_processed": len(ok),
        "photos_failed": len(diag.rows) - len(ok),
        "photos_with_bib_read": sum(bool(r.result.bibs) for r in ok),
        "photos_without_valid_bib_read": sum(not r.result.bibs for r in ok),
        "photo_bib_pairs_deduplicated": pairs,
        "ocr_lines": sum(g.lines for g in regions),
        "candidates_total": len(readings),
        "candidates_accepted": len(accepted),
        "unique_numeric_identifiers": len({x.numeric_identifier for x in accepted
                                            if x.numeric_identifier}),
        "ocr_calls": len([g for g in regions if g.status != "crop_too_small_for_ocr"]),
        "regions_processed": len(regions),
        "duplicate_readings_merged": len(accepted) - pairs,
        "accepted_after_confusion_fix": sum(x.corrected for x in accepted),
        "rejections_by_reason": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "regions_by_status": dict(sorted(region_status.items())),
        "photos_fallback_no_person": sum(r.result.fallback_used == "no_person" for r in ok),
        "photos_fallback_after_crops": sum(r.result.fallback_used == "no_valid_bib_in_crops" for r in ok),
        "timing_note": ("wall-clock per stage; per-photo statistics exclude the first photo "
                        "(warm-up), model loading and debug writing"),
        "time_per_photo_total_ms": _stats([r.total_ms for r in timed]),
        "time_per_photo_decode_ms": _stats([r.decode_ms for r in timed]),
        "time_per_photo_detection_ms": _stats([r.detect_ms for r in timed]),
        "time_per_photo_ocr_ms": _stats([r.crop_ocr_ms + r.fallback_ocr_ms for r in timed]),
        "time_per_ocr_call_person_crop_ms": _stats(crop_ms["person_body"]),
        "time_per_ocr_call_center_crop_ms": _stats(crop_ms["center_crop"]),
        "time_per_ocr_call_full_image_tile_ms": _stats(crop_ms["full_image"]),
        "first_photo_total_ms": ok[0].total_ms if ok else None,
    }


def _fmt(st):
    return "n/a" if not st else (f"mean {st['mean']:.0f} | median {st['median']:.0f} | p90 {st['p90']:.0f}"
                                 f" | min {st['min']:.0f} | max {st['max']:.0f} (n={st['n']})")


def report_text(diag: OCRDiagnostic) -> str:
    s, sy = summarize(diag), diag.system
    lines = [
        "# Runner Photo Finder - OCR diagnostic report (Phase 5)", "",
        "This report describes what detection + OCR produced on these photos.",
        "It does NOT measure OCR accuracy or search recall: those need the ground truth",
        "from `ground_truth_review.xlsx`. \"No valid bib read\" does not mean the photo",
        "has no bib.", "",
        f"- Started: {diag.started}", f"- Photo folder: `{diag.folder}`",
        f"- Output: `{diag.run_dir}`", "",
        "## System",
        f"- OS: {sy.get('os')}",
        f"- CPU: {sy.get('cpu')} ({sy.get('logical_cores')} logical cores), RAM: {sy.get('ram_gb')} GB",
        f"- Python {sy.get('python')}, torch {sy.get('torch', 'n/a')}, ultralytics "
        f"{sy.get('ultralytics', 'n/a')}, CUDA GPU: {sy.get('cuda_gpu') or 'none'}",
        f"- Model loading: {', '.join(f'{k} {v:.1f} s' for k, v in diag.model_load_s.items()) or 'n/a'}",
        "", "## Settings", *[f"- {k}: {v}" for k, v in diag.settings.items()], "",
        "## Results",
        f"- Photos found / selected / processed / failed: {s['photos_found_in_folder']} / "
        f"{s['photos_selected']} / {s['photos_processed']} / {s['photos_failed']}",
        f"- Photos with at least one valid bib read: {s['photos_with_bib_read']}",
        f"- Photos with no valid bib read: {s['photos_without_valid_bib_read']}",
        f"- Bibs per photo after deduplication (photo-bib pairs): {s['photo_bib_pairs_deduplicated']}",
        f"- OCR text lines: {s['ocr_lines']}; candidates: {s['candidates_total']} "
        f"(accepted {s['candidates_accepted']}, duplicates merged {s['duplicate_readings_merged']}, "
        f"accepted after letter/digit fix {s['accepted_after_confusion_fix']})",
        f"- Unique numeric identifiers among accepted readings: {s['unique_numeric_identifiers']}",
        f"- OCR calls / regions processed: {s['ocr_calls']} / {s['regions_processed']}",
        f"- Rejections by reason: {s['rejections_by_reason']}",
        f"- Regions by status: {s['regions_by_status']}",
        f"- Full-image fallback: {s['photos_fallback_no_person']} photos without a person, "
        f"{s['photos_fallback_after_crops']} photos after crops gave no bib", "",
        "## Processing time (milliseconds)",
        f"Method: {s['timing_note']}.",
        f"- Per photo, total: {_fmt(s['time_per_photo_total_ms'])}",
        f"- Per photo, decode: {_fmt(s['time_per_photo_decode_ms'])}",
        f"- Per photo, person detection: {_fmt(s['time_per_photo_detection_ms'])}",
        f"- Per photo, OCR (all crops + fallback): {_fmt(s['time_per_photo_ocr_ms'])}",
        f"- Per OCR call, person crop: {_fmt(s['time_per_ocr_call_person_crop_ms'])}",
        f"- Per OCR call, center crop: {_fmt(s['time_per_ocr_call_center_crop_ms'])}",
        f"- Per OCR call, full-image tile: {_fmt(s['time_per_ocr_call_full_image_tile_ms'])}",
        "- First photo (incl. warm-up): " +
        (f"{s['first_photo_total_ms']:.0f}" if s['first_photo_total_ms'] is not None else "n/a"),
        "", "## Photos",
        "| # | File | Bibs read (confidence, readings) | Status | Total ms |", "|---|---|---|---|---|",
    ]
    for r in diag.rows:
        if r.status == "ok":
            bibs = ", ".join(f"{b.bib} ({b.best_confidence:.2f}, x{b.readings})" for b in r.result.bibs)
            lines.append(f"| {r.order} | {Path(r.file).name} | {bibs or '-'} | {r.result.status} | "
                         f"{r.total_ms:.0f} |")
        else:
            lines.append(f"| {r.order} | {Path(r.file).name} | - | FAILED: {r.error} | |")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # BOM: Excel reads UTF-8
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def write_outputs(diag: OCRDiagnostic) -> None:
    run = Path(diag.run_dir)
    ok = [r for r in diag.rows if r.status == "ok"]

    _write_csv(run / "photos.csv",
               ["photo_no", "file", "status", "error", "width", "height", "exif_orientation",
                "persons_detected", "persons_kept", "regions_ocr", "fallback", "bibs",
                "bib_confidences", "candidates", "candidates_accepted", "decode_ms", "detection_ms",
                "crop_ocr_ms", "fallback_ocr_ms", "total_ms", "debug_write_ms", "overview_image"],
               [[r.order, r.file, r.result.status if r.result else r.status, r.error,
                 *(r.result.detection.image_size if r.result else ("", "")), r.orientation,
                 len(r.result.detection.persons) if r.result else "",
                 len(r.result.detection.kept_persons) if r.result else "",
                 len(r.result.regions) if r.result else "",
                 (r.result.fallback_used or "") if r.result else "",
                 ";".join(b.bib for b in r.result.bibs) if r.result else "",
                 ";".join(f"{b.best_confidence:.3f}" for b in r.result.bibs) if r.result else "",
                 len(r.result.readings) if r.result else "",
                 sum(x.accepted for x in r.result.readings) if r.result else "",
                 round(r.decode_ms, 1), round(r.detect_ms, 1), round(r.crop_ocr_ms, 1),
                 round(r.fallback_ocr_ms, 1), round(r.total_ms, 1), round(r.debug_write_ms, 1),
                 r.overview_image] for r in diag.rows])

    _write_csv(run / "photo_bibs.csv",
               ["photo_no", "file", "bib", "numeric_identifier", "best_confidence", "readings", "region_ids",
                "best_box_x", "best_box_y", "best_box_w", "best_box_h", "letter_digit_fix",
                "review_verdict", "corrected_bib", "review_notes"],
               [[r.order, r.file, b.bib, b.numeric_identifier or "", round(b.best_confidence, 4), b.readings,
                 ";".join(b.region_ids), *_xywh(b.best_box), b.corrected, "", "", ""]
                for r in ok for b in r.result.bibs])

    _write_csv(run / "readings.csv",
               ["photo_no", "file", "region_id", "region_source", "region_person_index",
                "line_index", "raw_text", "ocr_confidence", "line_x", "line_y", "line_w", "line_h",
                "token", "normalized", "candidate_bib", "numeric_identifier", "letter_digit_fix", "accepted",
                "rejection_reason"],
               [[r.order, r.file, x.region_id, x.region_source,
                 x.region_person_index if x.region_person_index is not None else "",
                 x.line_index, x.raw_text, round(x.ocr_confidence, 4), *_xywh(x.line_box),
                 x.token, x.normalized, x.bib or "", x.numeric_identifier or "", x.corrected,
                 x.accepted, x.reason or ""]
                for r in ok for x in r.result.readings])

    _write_csv(run / "regions.csv",
               ["photo_no", "file", "region_id", "source", "person_index", "det_confidence",
                "x", "y", "w", "h", "status", "ocr_lines", "ocr_ms", "error", "notes"],
               [[r.order, r.file, g.region_id, g.region.source,
                 g.region.person_index if g.region.person_index is not None else "",
                 round(g.region.det_confidence, 4), *_xywh(g.region.box), g.status, g.lines,
                 round(g.ocr_ms, 1), g.error, "; ".join(g.region.notes)]
                for r in ok for g in r.result.regions])

    review = []
    for r in diag.rows:
        res = r.result
        review.append([str(r.order), Path(r.file).name, os.path.relpath(r.file, diag.folder),
                       ";".join(b.bib for b in res.bibs) if res else "",
                       ";".join(f"{b.best_confidence:.2f}" for b in res.bibs) if res else "",
                       res.status if res else f"FAILED: {r.error}", "", "", "", "",
                       os.path.relpath(r.overview_image, diag.run_dir) if r.overview_image else ""])
    write_xlsx(run / "ground_truth_review.xlsx", REVIEW_HEADERS, review,
               widths=[9, 28, 36, 22, 18, 40, 24, 12, 22, 30, 50])
    _write_csv(run / "ground_truth_review.csv", REVIEW_HEADERS, review)
    (run / "REVIEW_INSTRUCTIONS.txt").write_text(REVIEW_INSTRUCTIONS, encoding="utf-8")
    (run / "report.md").write_text(report_text(diag), encoding="utf-8")
    (run / "report.json").write_text(json.dumps(
        {"summary": summarize(diag), "settings": diag.settings, "system": diag.system,
         "model_load_s": diag.model_load_s, "folder": diag.folder, "started": diag.started},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
