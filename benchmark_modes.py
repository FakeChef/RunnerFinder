"""Controlled five-mode OCR benchmark on the same photo selection."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from config import PROCESSING_MODES, get_config
from detection.person_detector import YoloPersonDetector
from ocr.bib_reader import PaddleOCREngine
from ocr.diagnostic import run_ocr_diagnostic, summarize


def run(folder: str, limit: int = 50, modes=PROCESSING_MODES) -> Path:
    cfg = get_config()
    cfg.ocr.enable_mkldnn = False
    weights = cfg.paths.models_dir / cfg.detection.model_name
    engine = PaddleOCREngine(cfg.ocr)
    detector = None
    results = []
    for mode in modes:
        cfg.detection.processing_mode = mode
        if mode != "center" and detector is None:
            detector = YoloPersonDetector(weights, cfg.detection.person_conf_threshold,
                                           cfg.detection.detection_image_size, cfg.detection.device)
        t = time.perf_counter()
        diag = run_ocr_diagnostic(folder, cfg, None if mode == "center" else detector,
                                  engine, limit=limit, spread=True, debug=False,
                                  model_load_s={}, progress=None)
        batch_s = time.perf_counter() - t
        summary = summarize(diag)
        timed = summary["time_per_photo_total_ms"]
        results.append({"mode": mode, "run_dir": diag.run_dir,
                        "batch_seconds": round(batch_s, 3), "summary": summary,
                        "warmup_photo_ms": summary["first_photo_total_ms"],
                        "timed_mean_photo_ms": timed.get("mean") if timed else None})

    out = cfg.paths.data_dir / "benchmarks" / f"lite_modes_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    usable = [r for r in results if r["timed_mean_photo_ms"] is not None]
    fastest = min(usable, key=lambda r: r["timed_mean_photo_ms"])["mode"] if usable else "n/a"
    lines = ["# Lite mode benchmark", "", f"- Folder: `{folder}`", f"- Limit: {limit}",
             "- Method: one first-photo warm-up is excluded from per-photo timed statistics;",
             "  all five modes use the same spread selection and CPU configuration.", "",
             "| Mode | Batch s | Warm-up ms | Mean ms/photo | OCR calls | Regions | Unique numeric IDs |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        s = r["summary"]
        lines.append(f"| {r['mode']} | {r['batch_seconds']:.1f} | {r['warmup_photo_ms'] or 0:.0f} | "
                     f"{r['timed_mean_photo_ms'] or 0:.0f} | {s['ocr_calls']} | {s['regions_processed']} | "
                     f"{s['unique_numeric_identifiers']} |")
    lines += ["", f"Fastest by mean timed photo: **{fastest}**.",
              "Accuracy is not claimed without manually completed ground truth."]
    (out / "COMPARISON_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark all Runner Photo Finder processing modes")
    p.add_argument("folder")
    p.add_argument("--limit", type=int, default=50)
    args = p.parse_args()
    out = run(args.folder, args.limit)
    print(f"Benchmark report: {out / 'COMPARISON_REPORT.md'}")
    print(f"Machine-readable results: {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
