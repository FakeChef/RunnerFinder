r"""
Command-line tools for large events. A browser refresh cannot interrupt them.

Scan a folder (Phase 3, no AI needed):

    .\.venv\Scripts\python.exe index_cli.py scan "D:\RacePhotos\Race" --event "2026 Race"

Use a separate database for testing:

    .\.venv\Scripts\python.exe index_cli.py scan "D:\RacePhotos\Test" --event Test --db data\test_scan.db

Scanning only reads the photos. Re-running it is safe and fast: unchanged
files are skipped, so an interrupted scan simply continues.

Detection diagnostic (Phase 4, needs the full install): process 20-50
photos spread across a folder and write debug images plus a report under
data\debug\detect_<date>_<time>\ :

    .\.venv\Scripts\python.exe index_cli.py detect "D:\RacePhotos\Sample" --limit 30

V0.2 SQLite index. Store OCR results per photo while the run goes on
(CSV outputs are written as before):

    .\.venv\Scripts\python.exe index_cli.py ocr "D:\RacePhotos\Race" --mode lite1 --event "2026 Race"

Import an existing diagnostic batch into an event (the batch is only read):

    .\.venv\Scripts\python.exe index_cli.py import-csv "data\debug\ocr_<date>_<time>" --event "2026 Race"

Both refuse to overwrite existing results or to mix different OCR settings
in one event; --replace does it on purpose.

Incremental analysis (V0.2 Phase 4): analyse a whole folder into an event.
Later runs only analyse new, changed and failed photos; an interrupted run
continues where it stopped (GPU: analyze_gpu.bat with the same arguments):

    .\.venv\Scripts\python.exe index_cli.py analyze "D:\RacePhotos\Race" --event "2026 Race"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

from config import CROP_PRESETS, PROCESSING_MODES, get_config
from database.database import get_or_create_event, init_db
from scanner.folder_sync import sync_folder
from utils.sysinfo import system_summary


def system_info() -> str:
    s = system_summary()
    return (f"{s['os']} | CPU: {s['cpu']} | logical cores: {s['logical_cores']} | "
            f"RAM: {s['ram_gb']} GB | Python {s['python']}")


def cmd_scan(args: argparse.Namespace) -> int:
    cfg = get_config()
    db_path = args.db or cfg.paths.database_file
    event_name = args.event or Path(args.folder).name
    conn = init_db(db_path)
    event_id = get_or_create_event(conn, event_name)

    print(f"System:   {system_info()}")
    print(f"Folder:   {os.path.abspath(args.folder)}")
    print(f"Event:    {event_name} (id {event_id})")
    print(f"Database: {os.path.abspath(db_path)}\n")

    last = [0.0]

    def progress(done: int, path: str) -> None:
        now = time.perf_counter()
        if now - last[0] > 1.0:                # print at most once per second
            print(f"\r  {done} files scanned ...", end="", flush=True)
            last[0] = now

    try:
        s = sync_folder(conn, event_id, args.folder, cfg.scan,
                        exclude_dirs=(cfg.paths.default_export_root,), progress=progress)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Progress was saved; run the same command again to continue.")
        return 130

    rate = s.found / s.elapsed_s if s.elapsed_s else 0
    print(f"\r  done.{' ' * 30}\n")
    print(f"Files found:        {s.found}")
    print(f"  new:              {s.new}")
    print(f"  changed:          {s.changed}")
    print(f"  unchanged:        {s.unchanged}")
    print(f"  moved/renamed:    {s.relinked}")
    print(f"  unreadable:       {s.unreadable}")
    print(f"Newly missing:      {s.newly_missing}")
    print(f"Wall-clock time:    {s.elapsed_s:.2f} s ({rate:.0f} files/s, whole scan incl. database)")

    rows = conn.execute(
        "SELECT * FROM photos WHERE event_id = ? AND file_status = 'ok' ORDER BY capture_time, file_name",
        (event_id,)).fetchall()
    sources = Counter(r["capture_time_source"] or "none" for r in rows)
    cameras = Counter(f'{r["camera_make"] or "?"} {r["camera_model"] or "?"} #{r["camera_serial"] or "?"}'
                      for r in rows)
    with_subsec = sum(1 for r in rows if r["capture_time"] and "." in r["capture_time"])
    rotated = sum(1 for r in rows if r["exif_orientation"] not in (None, 1))
    print(f"\nCapture time sources: {dict(sources)}")
    print(f"With sub-second time: {with_subsec} of {len(rows)}")
    print(f"Rotated by EXIF:      {rotated}")
    print("Cameras:")
    for cam, n in cameras.most_common():
        print(f"  {n:>6}  {cam}")
    print("\nFirst photos by capture time:")
    for r in rows[:5]:
        print(f"  {r['capture_time'] or '(no time)':<26} {r['width']}x{r['height']}  {r['file_path']}")
    if s.errors:
        print(f"\nProblems ({len(s.errors)}, first 20 shown):")
        for path, msg in s.errors[:20]:
            print(f"  {path}\n      {msg}")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    from detection.benchmark import report_text, run_detection_benchmark, summarize
    from detection.person_detector import YoloPersonDetector

    cfg = get_config()
    d = cfg.detection
    _apply_detection_args(cfg, args)
    cfg.validate()

    print(f"System: {system_info()}")
    weights = cfg.paths.models_dir / d.model_name
    print(f"Loading {weights.name} (downloaded once if missing) ...")
    t = time.perf_counter()
    detector = YoloPersonDetector(weights, d.person_conf_threshold, d.detection_image_size, d.device)
    load_s = time.perf_counter() - t

    def progress(i: int, n: int, path: str) -> None:
        print(f"  [{i}/{n}] {path}")

    try:
        report = run_detection_benchmark(args.folder, cfg, detector, limit=args.limit,
                                         spread=not args.first, debug=not args.no_debug,
                                         model_load_s=load_s, progress=progress)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    s = summarize(report)
    print()
    print(report_text(report).split("## Images")[0])
    print(f"Report:        {Path(report.run_dir) / 'report.md'}")
    print(f"Per-image CSV: {Path(report.run_dir) / 'images.csv'}")
    if not args.no_debug:
        print(f"Debug images:  {Path(report.run_dir) / 'images'}")
    return 0 if s["images_processed"] else 1


def _apply_detection_args(cfg, args) -> None:
    d = cfg.detection
    if args.crop_strategy:
        d.crop_strategy = args.crop_strategy
    if args.conf is not None:
        d.person_conf_threshold = args.conf
    if args.device:
        d.device = args.device
    if args.fallback:
        d.full_image_fallback = args.fallback


def cmd_ocr(args: argparse.Namespace) -> int:
    from detection.person_detector import YoloPersonDetector
    from ocr.bib_reader import PaddleOCREngine
    from ocr.candidates import load_registered_bibs
    from ocr.diagnostic import report_text, run_ocr_diagnostic, summarize

    cfg = get_config()
    cfg.detection.processing_mode = args.mode
    _apply_detection_args(cfg, args)
    if args.mkldnn:
        cfg.ocr.enable_mkldnn = True
    if args.ocr_device:
        cfg.ocr.device = args.ocr_device
    cfg.validate()
    registered = load_registered_bibs(args.registered_bibs, cfg.bib) if args.registered_bibs else None

    print(f"System: {system_info()}")
    print(f"PaddleOCR model cache: {os.environ.get('PADDLE_PDX_CACHE_HOME')}")
    d = cfg.detection
    detector = None
    load = {}
    if args.mode != "center":
        t = time.perf_counter()
        detector = YoloPersonDetector(cfg.paths.models_dir / d.model_name, d.person_conf_threshold,
                                      d.detection_image_size, d.device)
        load["yolo"] = time.perf_counter() - t
    print("Loading PaddleOCR (downloaded once if missing) ...")
    t = time.perf_counter()
    from utils.gpu_runtime import GPUUnavailable
    try:
        engine = PaddleOCREngine(cfg.ocr)
    except GPUUnavailable as exc:
        print(f"ERROR: OCR device '{cfg.ocr.device}' cannot be used: {exc}")
        return 4
    load["paddleocr"] = time.perf_counter() - t
    print(f"OCR device: requested {cfg.ocr.device} -> using {engine.device}"
          + (f" ({engine.device_info.get('name')}, {engine.device_info.get('memory_mib')} MiB, "
             f"CUDA {engine.device_info.get('cuda')}, cuDNN {engine.device_info.get('cudnn')}, "
             f"Paddle {engine.device_info.get('paddle')})" if engine.device_info else ""))
    if engine.device_note:
        print(f"WARNING: {engine.device_note}")

    def progress(i: int, n: int, path: str) -> None:
        print(f"  [{i}/{n}] {path}")

    store = None
    if args.event:
        from database.results_store import OCRRunStore
        conn = init_db(args.db or cfg.paths.database_file)
        store = OCRRunStore(conn, get_or_create_event(conn, args.event), replace=args.replace)
        print(f"Saving each photo to event '{args.event}' in {os.path.abspath(args.db or cfg.paths.database_file)}")
    try:
        diag = run_ocr_diagnostic(args.folder, cfg, detector, engine, limit=args.limit,
                                  spread=not args.first, debug=not args.no_debug,
                                  registered=registered, model_load_s=load, progress=progress,
                                  store=store)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    except KeyboardInterrupt:
        if store:
            print(f"\nInterrupted. {store.saved} photo(s) were saved to the database (run {store.run_id}).")
        return 130
    except Exception as exc:
        from database.results_store import ResultConflict
        if isinstance(exc, ResultConflict):
            print(f"ERROR: {exc}")
            return 3
        raise
    if store:
        for w in store.warnings:
            print(f"WARNING (--replace): {w}")
        print(f"Database: run {store.run_id}, {store.saved} photo(s) saved, settings hash {store.hash}")
    print()
    print(report_text(diag).split("## Photos")[0])
    run = Path(diag.run_dir)
    for name in ("report.md", "photos.csv", "photo_bibs.csv", "readings.csv", "regions.csv",
                 "ground_truth_review.xlsx", "REVIEW_INSTRUCTIONS.txt"):
        print(f"  {run / name}")
    return 0 if summarize(diag)["photos_processed"] else 1


def cmd_import_csv(args: argparse.Namespace) -> int:
    from database.csv_import import ResultConflict, import_batch

    cfg = get_config()
    db_path = args.db or cfg.paths.database_file
    conn = init_db(db_path)
    print(f"Batch:    {os.path.abspath(args.batch)}")
    print(f"Event:    {args.event}")
    print(f"Database: {os.path.abspath(db_path)}\n")
    try:
        s = import_batch(conn, args.batch, args.event, replace=args.replace)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    except ResultConflict as exc:
        print(f"ERROR: {exc}")
        return 3
    if s.already_imported:
        print("This batch was already imported into this event. Nothing changed.")
        return 0
    for w in s.warnings:
        print(f"WARNING (--replace): {w}")
    for n in s.notes:
        print(f"NOTE: {n}")
    print(f"Run id:            {s.run_id} (settings hash {s.settings_hash})")
    print(f"Photos:              {s.photos}")
    print(f"  with bibs:         {s.photos_with_bibs}")
    print(f"  failed in run:     {s.photos_failed}")
    print(f"  file now missing:  {s.photo_files_missing}")
    print(f"Photo-bib pairs:     {s.pairs}")
    print(f"Feedback from review_overrides.csv: {s.feedback_added} added, {s.overrides_ignored} ignored")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from database.results_store import ResultConflict
    from indexing.analyze import analyze_folder
    from utils.gpu_runtime import GPUUnavailable

    cfg = get_config()
    cfg.detection.processing_mode = args.mode
    if args.ocr_device:
        cfg.ocr.device = args.ocr_device
    cfg.validate()
    db_path = args.db or cfg.paths.database_file
    conn = init_db(db_path)
    print(f"System:   {system_info()}")
    print(f"Folder:   {os.path.abspath(args.folder)}")
    print(f"Event:    {args.event}")
    print(f"Database: {os.path.abspath(db_path)}")
    print(f"Mode:     {args.mode}, OCR device requested: {cfg.ocr.device}\n")

    def load_models():
        from detection.person_detector import YoloPersonDetector
        from ocr.bib_reader import PaddleOCREngine
        d = cfg.detection
        detector = None
        if args.mode != "center":
            detector = YoloPersonDetector(cfg.paths.models_dir / d.model_name, d.person_conf_threshold,
                                          d.detection_image_size, d.device)
        print("Loading PaddleOCR ...")
        engine = PaddleOCREngine(cfg.ocr)
        print(f"OCR device: requested {cfg.ocr.device} -> using {engine.device}"
              + (f" ({engine.device_info.get('name')}, CUDA {engine.device_info.get('cuda')})"
                 if engine.device_info else ""))
        if engine.device_note:
            print(f"WARNING: {engine.device_note}")
        return detector, engine

    last = [0.0]

    def scan_progress(done: int, path: str) -> None:
        now = time.perf_counter()
        if now - last[0] > 1.0:
            print(f"\r  scanning: {done} files ...", end="", flush=True)
            last[0] = now

    def progress(i: int, n: int, path: str) -> None:
        print(f"  [{i}/{n}] {path}")

    try:
        s = analyze_folder(conn, args.folder, args.event, cfg, load_models, replace=args.replace,
                           limit=args.limit, debug=args.debug, progress=progress,
                           scan_progress=scan_progress)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    except ResultConflict as exc:
        print(f"ERROR: {exc}")
        return 3
    except GPUUnavailable as exc:
        print(f"ERROR: OCR device '{cfg.ocr.device}' cannot be used: {exc}")
        return 4

    sc, sel = s.scan, s.selection
    print(f"\r  scan done.{' ' * 30}")
    print(f"Scan:      {sc.found} files ({sc.new} new, {sc.changed} changed, {sc.relinked} moved/renamed, "
          f"{sc.unreadable} unreadable, {sc.newly_missing} newly missing)")
    if s.stale_runs_cancelled:
        print(f"Recovered: {s.stale_runs_cancelled} earlier run(s) had stopped unexpectedly; their "
              f"finished photos are kept")
    print(f"To analyse: {len(sel.new)} new, {len(sel.changed)} changed, {len(sel.failed)} failed before"
          + (f", {len(sel.other_settings)} with other OCR settings (--replace)" if args.replace else "")
          + f"; skipped: {sel.done} already analysed, {sel.unreadable} unreadable, {sel.missing} missing")
    if s.to_process == 0:
        print("Nothing to analyse. The index is up to date.")
        return 0
    for w in s.warnings:
        print(f"WARNING (--replace): {w}")
    rate = s.processed / s.elapsed_s if s.elapsed_s else 0
    print(f"Analysed:  {s.processed} of {s.to_process} photo(s) in {s.elapsed_s:.0f} s "
          f"(run {s.run_id}, device {s.device}, settings hash {s.settings_hash}); failed: {s.failed_now}")
    if cfg.thumbnails.generate_during_analyze:
        print(f"Thumbnails: {s.thumbnails} available in {cfg.paths.thumbnail_cache}")
    if s.interrupted:
        print("Interrupted. Finished photos are saved; run the same command again to continue.")
        return 130
    if s.run_dir:
        print(f"Diagnostic CSV files: {s.run_dir}")
    return 0


def cmd_thumbs(args: argparse.Namespace) -> int:
    """Create missing Search-page thumbnails for an event (originals are only read)."""
    from utils.thumbnails import ensure_thumbnail, existing_thumbnail

    cfg = get_config()
    conn = init_db(args.db or cfg.paths.database_file)
    ev = conn.execute("SELECT id FROM events WHERE name = ?", (args.event,)).fetchone()
    if ev is None:
        print(f"ERROR: no event named '{args.event}'")
        return 2
    rows = conn.execute("SELECT file_path, fingerprint FROM photos WHERE event_id = ? AND "
                        "file_status = 'ok' AND fingerprint IS NOT NULL", (ev["id"],)).fetchall()
    cache = cfg.paths.thumbnail_cache
    print(f"Event: {args.event} ({len(rows)} photos)\nThumbnails: {cache}")
    t0, last = time.perf_counter(), 0.0
    existing = created = failed = 0
    for i, r in enumerate(rows, start=1):
        if existing_thumbnail(cache, r["fingerprint"]):
            existing += 1
        elif ensure_thumbnail(r["file_path"], cache, r["fingerprint"], cfg.thumbnails):
            created += 1
        else:
            failed += 1
        if time.perf_counter() - last > 1.0:
            print(f"\r  {i}/{len(rows)} ...", end="", flush=True)
            last = time.perf_counter()
    print(f"\r  done.{' ' * 20}\nAlready there: {existing}, created: {created}, failed: {failed}, "
          f"time: {time.perf_counter() - t0:.0f} s")
    return 0 if failed == 0 else 1


def cmd_filter_bibs(args: argparse.Namespace) -> int:
    """Bib filter on ONE event's stored candidates (no OCR). Dry run unless --apply."""
    from indexing.filter_bibs import rescore_event, restore_event_filter, summary_json
    from ocr.bib_filter import BibCandidateFilter

    cfg = get_config()
    db_path = args.db or cfg.paths.database_file
    rules_path = args.rules or cfg.paths.bib_rules_file
    checks = tuple(q.strip() for q in (args.check_bibs or "").split(",") if q.strip())
    try:
        conn = init_db(db_path, video_extension=False)       # never migrates as a side effect
        if args.restore_from:
            s = restore_event_filter(conn, db_path, args.event, args.restore_from, apply=args.apply,
                                     check_bibs=checks)
        else:
            bib_filter = BibCandidateFilter.load(rules_path)
            s = rescore_event(conn, db_path, args.event, bib_filter, apply=args.apply, clear=args.clear,
                              check_bibs=checks)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        return 3
    mode = "APPLIED" if s.applied else "DRY RUN (nothing changed; add --apply to store)"
    what = (f"restore filter results from {s.restored_from}" if s.restored_from else
            "remove all filter results" if s.cleared else str(rules_path))
    print(f"Event:   {s.event_name}\nRules:   {what} [{s.rules}]\nMode:    {mode}")
    print(f"Candidates stored:          {s.rows} (none is deleted)")
    print(f"Status before:              {dict(s.status_before)}")
    print(f"Status after:               {dict(s.status_after)}")
    print(f"Shown in search:            {s.visible_before} -> {s.visible_after} "
          f"({s.newly_hidden} newly hidden, {s.newly_visible} shown again)")
    print(f"Photos with a confirmed bib:      {s.photos_confirmed_before} -> {s.photos_confirmed_after} "
          f"of {s.photos}")
    print(f"Photos findable (incl. low conf.): {s.photos_findable_before} -> {s.photos_findable_after} "
          f"of {s.photos}")
    print(f"Distinct confirmed numbers: {s.numbers_before} -> {s.numbers_after}")
    for reason, n in s.reasons.most_common():
        print(f"  not confirmed: {reason:22} {n:6}   e.g. {'; '.join(s.examples.get(reason, [])[:2])}")
    for q, res in s.checks.items():
        b, a = res.get("before", {}), res.get("after", {})
        print(f"  search {q!r:10} photos {b.get('photos')} -> {a.get('photos')} "
              f"(confirmed {b.get('confirmed')} -> {a.get('confirmed')}, low confidence "
              f"{b.get('low_confidence')} -> {a.get('low_confidence')}, hidden {b.get('hidden')} -> {a.get('hidden')})")
    if s.backup:
        print(f"Backup before the change:   {s.backup} (validated: {s.backup_validated})")
        print(f"Unchanged after the change: {s.integrity}")
    if args.json_out:
        Path(args.json_out).write_text(summary_json(s), encoding="utf-8")
    return 0


def cmd_gpu_check(args: argparse.Namespace) -> int:
    """Check that PaddlePaddle can use the NVIDIA GPU, without loading any OCR model."""
    from utils.gpu_runtime import GPUUnavailable, gpu_status, registered_dirs
    print(f"Python:           {sys.executable}")
    print(f"CUDA DLL folders: {len(registered_dirs)} registered")
    for d in registered_dirs:
        print(f"  {d}")
    try:
        info = gpu_status(0)
    except GPUUnavailable as exc:
        print(f"GPU OCR: NOT available - {exc}")
        return 1
    print(f"GPU OCR: available on {info['device']}: {info['name']}, {info['memory_mib']} MiB, "
          f"compute {info['compute_capability']}, CUDA {info['cuda']}, cuDNN {info['cudnn']}, "
          f"Paddle {info['paddle']}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from benchmark_modes import run
    out = run(args.folder, args.limit)
    print(f"Comparison report: {out / 'COMPARISON_REPORT.md'}")
    print(f"Machine-readable results: {out / 'results.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Runner Photo Finder command-line tools")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("scan", help="Scan a photo folder into the database (no AI)")
    p.add_argument("folder", help="Folder containing race photos")
    p.add_argument("--event", help="Event name (default: folder name)")
    p.add_argument("--db", help="Database file (default: data/runner_photo_finder.db)")
    p.set_defaults(func=cmd_scan)

    d = sub.add_parser("detect", help="Detection diagnostic on sample photos (needs AI stack)")
    d.add_argument("folder", help="Folder with sample race photos (searched recursively)")
    d.add_argument("--limit", type=int, default=30,
                   help="number of photos, spread evenly over the folder (default 30)")
    d.add_argument("--first", action="store_true", help="take the first N photos instead")
    d.add_argument("--no-debug", action="store_true", help="do not write debug images")
    d.add_argument("--crop-strategy", choices=[*CROP_PRESETS, "custom"])
    d.add_argument("--conf", type=float, help="person confidence threshold (default 0.25)")
    d.add_argument("--device", help='"cpu" (default), "auto" or e.g. "cuda:0"')
    d.add_argument("--fallback", choices=["never", "no_person", "no_valid_bib"])
    d.set_defaults(func=cmd_detect)

    o = sub.add_parser("ocr", help="Detection + OCR diagnostic on sample photos (needs AI stack)")
    o.add_argument("folder", help="Folder with sample race photos (searched recursively)")
    o.add_argument("--limit", type=int, default=50,
                   help="number of photos, spread evenly over the folder (default 50)")
    o.add_argument("--first", action="store_true", help="take the first N photos instead")
    o.add_argument("--no-debug", action="store_true", help="do not write debug images")
    o.add_argument("--registered-bibs", help="text file with one registered bib per line (optional)")
    o.add_argument("--crop-strategy", choices=[*CROP_PRESETS, "custom"])
    o.add_argument("--conf", type=float, help="person confidence threshold (default 0.25)")
    o.add_argument("--device", help='YOLO device: "cpu" (default), "auto" or e.g. "cuda:0"')
    o.add_argument("--fallback", choices=["never", "no_person", "no_valid_bib"])
    o.add_argument("--mode", choices=PROCESSING_MODES, default="full",
                   help="processing mode: center, lite1, lite2, lite3 or full")
    o.add_argument("--mkldnn", action="store_true",
                   help="enable oneDNN in PaddleOCR (off by default; not verified on Windows)")
    o.add_argument("--ocr-device", choices=["cpu", "gpu", "auto"],
                   help='OCR device: "cpu" (default), "gpu" (GPU environment; error if unusable) '
                        'or "auto" (GPU if usable, otherwise CPU)')
    o.add_argument("--event", help="also save each photo's bibs to the SQLite index under this event")
    o.add_argument("--db", help="Database file (default: data/runner_photo_finder.db)")
    o.add_argument("--replace", action="store_true",
                   help="allow overwriting existing results / mixing OCR settings in the event")
    o.set_defaults(func=cmd_ocr)

    i = sub.add_parser("import-csv", help="Import an OCR diagnostic batch folder into the SQLite index")
    i.add_argument("batch", help=r"batch folder, e.g. data\debug\ocr_20261014_081736")
    i.add_argument("--event", required=True, help="event name (created if it does not exist)")
    i.add_argument("--db", help="Database file (default: data/runner_photo_finder.db)")
    i.add_argument("--replace", action="store_true",
                   help="allow overwriting existing results / mixing OCR settings in the event")
    i.set_defaults(func=cmd_import_csv)
    a = sub.add_parser("analyze", help="Incremental analysis of a photo folder into an event "
                                       "(only new, changed and failed photos)")
    a.add_argument("folder", help="Photo folder (searched recursively)")
    a.add_argument("--event", required=True, help="event name (created if it does not exist)")
    a.add_argument("--mode", choices=PROCESSING_MODES, default="lite1",
                   help="processing mode (default lite1, as used for the accepted test batches)")
    a.add_argument("--ocr-device", choices=["cpu", "gpu", "auto"], help='OCR device (default "cpu")')
    a.add_argument("--db", help="Database file (default: data/runner_photo_finder.db)")
    a.add_argument("--limit", type=int, help="analyse at most N photos in this run")
    a.add_argument("--debug", action="store_true", help="also write debug images (off by default)")
    a.add_argument("--replace", action="store_true",
                   help="also re-analyse photos done with other OCR settings in this event")
    a.set_defaults(func=cmd_analyze)

    th = sub.add_parser("thumbs", help="Create missing Search-page thumbnails for an event")
    th.add_argument("--event", required=True, help="event name")
    th.add_argument("--db", help="Database file (default: data/runner_photo_finder.db)")
    th.set_defaults(func=cmd_thumbs)

    fb = sub.add_parser("filter-bibs", help="Noise filter on stored bib candidates of an event "
                                            "(dry run unless --apply; no OCR)")
    fb.add_argument("--event", required=True, help="event name")
    fb.add_argument("--rules", help="rules file (default: config/bib_rules.yaml)")
    fb.add_argument("--apply", action="store_true",
                    help="store the results (a database backup is made first); default is a dry run")
    fb.add_argument("--clear", action="store_true",
                    help="remove all filter results of the event (with --apply): back to V0.2 behaviour")
    fb.add_argument("--db", help="Database file (default: data/runner_photo_finder.db)")
    fb.add_argument("--check-bibs", help="comma-separated searches to compare before/after, e.g. A0175,0175,175")
    fb.add_argument("--restore-from", help="backup file: put back this event's filter results from it "
                                          "(validated first; dry run unless --apply)")
    fb.add_argument("--json-out", help="also write the summary as JSON to this file")
    fb.set_defaults(func=cmd_filter_bibs)

    g = sub.add_parser("gpu-check", help="Check that PaddleOCR can use the NVIDIA GPU (no models loaded)")
    g.set_defaults(func=cmd_gpu_check)

    b = sub.add_parser("benchmark", help="Benchmark all five processing modes")
    b.add_argument("folder", help="Folder with sample race photos")
    b.add_argument("--limit", type=int, default=50,
                   help="number of photos, spread evenly over the folder (default 50)")
    b.set_defaults(func=cmd_benchmark)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
