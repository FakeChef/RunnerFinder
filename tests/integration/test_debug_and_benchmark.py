"""Debug images and the detection benchmark, with the colour-blob stand-in detector."""
import builtins
import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from detection.benchmark import run_detection_benchmark, summarize
from detection.bib_detector import RegionFinder
from detection.debug_viz import ensure_inside, write_debug
from scanner.photo_scanner import load_image_with_orientation
from tests.helpers import ColorBlobDetector, draw_people, make_test_config

RED, BLUE = (230, 20, 20), (20, 40, 230)


def snapshot(folder: Path):
    return {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(folder.rglob("*")) if p.is_file()}


@pytest.fixture
def photos(tmp_path):
    f = tmp_path / "photos"
    draw_people(f / "01_two_runners.jpg", (3000, 2000),
                [(RED, (300, 300, 800, 1900)), (BLUE, (1800, 200, 2300, 1950))])
    draw_people(f / "02_rotated.jpg", (2000, 3000), [(RED, (300, 300, 1700, 900))], orientation=6)
    draw_people(f / "03_no_person.jpg", (2000, 1333), [])
    draw_people(f / "sub" / "04_cut_at_bottom.png", (1600, 1200),
                [(BLUE, (500, 400, 800, 1200))], fmt="PNG")
    (f / "05_corrupt.jpg").write_bytes(b"\xff\xd8 broken")
    draw_people(f / "终点 06.jpg", (1600, 1200), [(RED, (100, 100, 500, 1100))])
    return f


def test_write_debug_outputs(tmp_path, photos):
    cfg = make_test_config(tmp_path)
    src = photos / "01_two_runners.jpg"
    image, orientation = load_image_with_orientation(src)
    result = RegionFinder(cfg.detection, ColorBlobDetector([RED, BLUE])).find(image)
    out = cfg.paths.debug_dir / "one"
    paths = write_debug(out, cfg.paths.data_dir, "001_x", str(src), image, result,
                        {"total_ms": 12.3}, orientation, max_side=1000)

    with Image.open(paths["overview"]) as overview:
        assert overview.width == 1000 and overview.height > 667   # downscaled + header strip
    assert len(paths["crops"]) == 2
    for crop_path, region in zip(paths["crops"], result.regions):
        with Image.open(crop_path) as crop:
            assert crop.size == (region.box.width, region.box.height)
    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert data["file"] == str(src) and data["image_size"] == [3000, 2000]
    assert len(data["persons"]) == 2 and data["persons"][0]["kept"]
    assert all(Path(p).resolve().is_relative_to(cfg.paths.data_dir.resolve())
               for p in [paths["overview"], paths["json"], *paths["crops"]])


def test_debug_output_outside_data_dir_is_refused(tmp_path, photos):
    cfg = make_test_config(tmp_path)
    with pytest.raises(ValueError):
        ensure_inside(photos / "debug", cfg.paths.data_dir)
    with pytest.raises(ValueError):
        ensure_inside(cfg.paths.data_dir / ".." / "elsewhere", cfg.paths.data_dir)


def test_benchmark_report(tmp_path, photos):
    cfg = make_test_config(tmp_path)
    report = run_detection_benchmark(photos, cfg, ColorBlobDetector([RED, BLUE]), limit=50)
    s = summarize(report)
    assert s["images_found_in_folder"] == 6
    assert (s["images_processed"], s["images_failed"]) == (5, 1)
    assert s["images_with_person_detected"] == 4
    assert s["person_detections_total"] == 5
    assert s["ocr_regions_full_image_tiles"] == 1               # the no-person photo
    assert s["crops_at_image_edge"] >= 1                         # cut at the bottom
    failed = [r for r in report.rows if r.status == "failed"]
    assert failed[0].file.endswith("05_corrupt.jpg") and failed[0].error
    ok = [r for r in report.rows if r.status == "ok"]
    assert all(r.total_ms > 0 and Path(r.overview_image).exists() for r in ok)
    rotated = next(r for r in ok if "rotated" in r.file)
    assert (rotated.width, rotated.height, rotated.orientation) == (3000, 2000, 6)

    run = Path(report.run_dir)
    md = (run / "report.md").read_text(encoding="utf-8")
    assert "does NOT measure bib recognition accuracy" in md
    assert "Images processed: 5" in md and "Failed / unreadable images: 1" in md
    with open(run / "images.csv", encoding="utf-8-sig") as f:
        assert len(list(csv.DictReader(f))) == 6
    assert json.loads((run / "report.json").read_text(encoding="utf-8"))["summary"]["images_processed"] == 5


def test_benchmark_never_writes_outside_data_dir(tmp_path, photos, monkeypatch):
    cfg = make_test_config(tmp_path)
    before = snapshot(photos)
    data_root = cfg.paths.data_dir.resolve()
    real_open = builtins.open

    def guarded_open(file, mode="r", *a, **kw):
        if set(mode) & set("wax+"):
            assert Path(file).resolve().is_relative_to(data_root), f"write outside data dir: {file}"
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", guarded_open)
    run_detection_benchmark(photos, cfg, ColorBlobDetector([RED, BLUE]), limit=50)
    monkeypatch.undo()
    assert snapshot(photos) == before                 # originals: same bytes, same mtime


def test_benchmark_limit_spreads_and_debug_can_be_off(tmp_path, photos):
    cfg = make_test_config(tmp_path)
    from scanner.photo_scanner import iter_photo_files
    order = [p for p, _ in iter_photo_files(photos)]
    report = run_detection_benchmark(photos, cfg, ColorBlobDetector([RED]), limit=2, debug=False)
    # 2 of 6: the first and the last photo in scan order, not the first two
    assert [r.file for r in report.rows] == [order[0], order[-1]]
    assert not (Path(report.run_dir) / "images").exists()


def test_benchmark_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_detection_benchmark(tmp_path / "nope", make_test_config(tmp_path), ColorBlobDetector([RED]))


def test_data_folder_inside_photo_folder_is_refused(tmp_path, photos):
    from config import AppConfig, PathsConfig
    data = photos / "data"
    cfg = AppConfig(paths=PathsConfig(data_dir=data, debug_dir=data / "debug"))
    with pytest.raises(ValueError):
        run_detection_benchmark(photos, cfg, ColorBlobDetector([RED]))
