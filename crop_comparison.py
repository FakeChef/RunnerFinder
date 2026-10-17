"""Run the Lite Top 1 body-vs-torso comparison on one identical photo set."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from config import get_config
from detection.person_detector import YoloPersonDetector
from ocr.bib_reader import PaddleOCREngine
from ocr.diagnostic import run_ocr_diagnostic, summarize


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {r["file"]: r for r in csv.DictReader(fh)}


def _bib_rows(path: Path) -> dict[str, set[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        out: dict[str, set[str]] = {}
        for r in csv.DictReader(fh):
            out.setdefault(r["file"], set()).add(r["bib"])
        return out


def _crop_dimensions(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    widths = [int(r["w"]) for r in rows if r.get("w")]
    heights = [int(r["h"]) for r in rows if r.get("h")]
    return {"regions": len(rows), "mean_w": sum(widths) / len(widths) if widths else None,
            "mean_h": sum(heights) / len(heights) if heights else None,
            "min_w": min(widths) if widths else None, "max_w": max(widths) if widths else None,
            "min_h": min(heights) if heights else None, "max_h": max(heights) if heights else None}


def compare(folder: str | Path, limit: int = 50) -> Path:
    cfg = get_config()
    cfg.detection.processing_mode = "lite1"
    cfg.detection.full_image_fallback = "no_valid_bib"
    cfg.validate()
    root = cfg.paths.data_dir / "benchmarks" / f"lite1_body_vs_torso_{time.strftime('%Y%m%d_%H%M%S')}"
    root.mkdir(parents=True, exist_ok=True)

    detector = YoloPersonDetector(cfg.paths.models_dir / cfg.detection.model_name,
                                  cfg.detection.person_conf_threshold,
                                  cfg.detection.detection_image_size, cfg.detection.device)
    engine = PaddleOCREngine(cfg.ocr)
    runs = {}
    for strategy in ("body", "torso"):
        cfg.detection.crop_strategy = strategy
        started = time.perf_counter()
        diag = run_ocr_diagnostic(folder, cfg, detector, engine, limit=limit, spread=True,
                                   debug=True, model_load_s={}, progress=None)
        runs[strategy] = {"run_dir": diag.run_dir, "summary": summarize(diag),
                          "crop_dimensions": _crop_dimensions(Path(diag.run_dir) / "regions.csv"),
                          "wall_s": time.perf_counter() - started}

    body = _bib_rows(Path(runs["body"]["run_dir"]) / "photo_bibs.csv")
    torso = _bib_rows(Path(runs["torso"]["run_dir"]) / "photo_bibs.csv")
    all_files = sorted(set(body) | set(torso), key=str.lower)
    differences = root / "candidate_differences.csv"
    with differences.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "body_bibs", "torso_bibs", "body_only", "torso_only"])
        for file in all_files:
            b, t = body.get(file, set()), torso.get(file, set())
            if b != t:
                w.writerow([file, ";".join(sorted(b)), ";".join(sorted(t)),
                            ";".join(sorted(b - t)), ";".join(sorted(t - b))])

    payload = {"folder": os.path.abspath(os.fspath(folder)), "limit": limit,
               "accuracy_not_evaluated": True, "runs": runs,
               "difference_file": str(differences),
               "note": "Candidate differences require manual review; no ground-truth accuracy claim."}
    (root / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Lite Top 1 body vs torso crop comparison", "",
             "Same folder, spread selection, Lite Top 1, same detector/OCR settings.",
             "This is not an accuracy evaluation: no verified ground truth was supplied.", "",
             "| strategy | run | photos | total mean ms | OCR mean ms | candidates | accepted | pairs | OCR calls | crop dimensions (mean w x h; min-max) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for strategy, item in runs.items():
        s = item["summary"]
        d = item["crop_dimensions"]
        dims = f"{d['mean_w']:.1f} x {d['mean_h']:.1f}; w {d['min_w']}-{d['max_w']}, h {d['min_h']}-{d['max_h']}"
        lines.append(f"| {strategy} | `{item['run_dir']}` | {s['photos_processed']} | "
                     f"{s['time_per_photo_total_ms']['mean']:.1f} | {s['time_per_photo_ocr_ms']['mean']:.1f} | "
                     f"{s['candidates_total']} | {s['candidates_accepted']} | "
                     f"{s['photo_bib_pairs_deduplicated']} | {s['ocr_calls']} | {dims} |")
    lines += ["", f"Candidate differences: `{differences}`",
              "Each run directory preserves `readings.csv`, `regions.csv`, `photo_bibs.csv`, "
              "crop dimensions, OCR times and debug images."]
    (root / "COMPARISON_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--limit", type=int, default=50)
    args = p.parse_args()
    print(compare(args.folder, args.limit))
