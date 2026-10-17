"""Phase 4 acceptance: incremental and resumable analysis (stand-in detector and OCR)."""
import pytest

from database import database as db
from database.results_store import ResultConflict
from indexing.analyze import analyze_folder
from ocr import diagnostic
from tests.helpers import ColorBlobDetector, ColorTextOCR, draw_people, make_test_config

RED = (230, 20, 20)
CYAN, ORANGE = (0, 200, 200), (250, 140, 0)


def photo(path, bib_color=CYAN, shade=0):
    """A runner (red box) with a bib (colour patch); `shade` changes the content only."""
    return draw_people(path, (1200, 800), [(RED, (200, 100, 500, 700)), (bib_color, (300, 300, 360, 340))],
                       background=(90 + shade, 110, 90))


class Models:
    """load_models() for analyze_folder; counts how often models were loaded."""

    def __init__(self):
        self.loads = 0

    def __call__(self):
        self.loads += 1
        return ColorBlobDetector([RED]), ColorTextOCR({CYAN: ("00123", 0.95), ORANGE: ("UT001", 0.91)})


@pytest.fixture
def env(tmp_path):
    folder = tmp_path / "photos"
    for i in range(12):
        photo(folder / f"run_{i:02d}.jpg")
    cfg = make_test_config(tmp_path, processing_mode="lite1")
    conn = db.init_db(cfg.paths.database_file)
    yield folder, cfg, conn
    conn.close()


def analyze(env, **kw):
    folder, cfg, conn = env
    models = kw.pop("models", None) or Models()
    s = analyze_folder(conn, folder, kw.pop("event", "Race"), cfg, models, **kw)
    s.models = models
    return s


def rows(conn):
    """Every stored result row, including ids: identical means untouched."""
    return [tuple(r) for r in conn.execute(
        "SELECT d.id, p.file_name, d.bib_number, d.run_id FROM bib_detection d "
        "JOIN photos p ON p.id = d.photo_id ORDER BY d.id")]


def status(conn):
    return dict(conn.execute("SELECT analysis_status, COUNT(*) FROM photos GROUP BY 1").fetchall())


def test_first_run_processes_all_second_run_zero(env):
    _, _, conn = env
    first = analyze(env)
    assert (first.to_process, first.processed, len(first.selection.new)) == (12, 12, 12)
    assert status(conn) == {"done": 12} and len(rows(conn)) == 12
    before = rows(conn)

    second = analyze(env)
    assert (second.to_process, second.processed, second.selection.done) == (0, 0, 12)
    assert second.models.loads == 0                       # nothing to do: models not even loaded
    assert rows(conn) == before                            # existing results untouched


def test_ten_new_photos_only_ten_processed(env):
    folder, _, conn = env
    analyze(env)
    before = rows(conn)
    for i in range(10):
        photo(folder / "later" / f"new_{i:02d}.jpg")
    s = analyze(env)
    assert (s.scan.new, len(s.selection.new), s.processed, s.selection.done) == (10, 10, 10, 12)
    assert rows(conn)[:12] == before and len(rows(conn)) == 22


def test_modified_photo_only_that_one_processed(env):
    folder, _, conn = env
    analyze(env)
    before = {r[1]: r for r in rows(conn)}
    photo(folder / "run_05.jpg", bib_color=ORANGE, shade=7)   # edited: other bib, other content
    s = analyze(env)
    assert (s.scan.changed, len(s.selection.changed), s.processed) == (1, 1, 1)
    after = {r[1]: r for r in rows(conn)}
    assert after["run_05.jpg"][2] == "UT001" and after["run_05.jpg"][3] == s.run_id
    assert {k: v for k, v in after.items() if k != "run_05.jpg"} == \
        {k: v for k, v in before.items() if k != "run_05.jpg"}


def test_interrupt_then_restart_continues_without_duplicates(env):
    _, _, conn = env

    def stop_before_fifth(i, n, path):
        if i == 5:
            raise KeyboardInterrupt
    first = analyze(env, progress=stop_before_fifth)
    assert first.interrupted and first.processed == 4
    assert status(conn) == {"done": 4, "pending": 8}
    assert conn.execute("SELECT status FROM analysis_runs WHERE id = ?", (first.run_id,)).fetchone()[0] \
        == "cancelled"

    second = analyze(env)
    assert (second.to_process, second.processed, second.selection.done) == (8, 8, 4)
    assert status(conn) == {"done": 12}
    pairs = conn.execute("SELECT photo_id, bib_number FROM bib_detection").fetchall()
    assert len(pairs) == 12 and len(set(map(tuple, pairs))) == 12        # one result per photo
    assert analyze(env).to_process == 0


def test_crash_or_power_loss_is_recovered(env):
    """A crash leaves the run 'running' and the photo in progress uncommitted."""
    _, _, conn = env

    def crash_at_seventh(i, n, path):
        if i == 7:
            raise SystemExit("simulated crash")     # not caught: like the process dying
    with pytest.raises(SystemExit):
        analyze(env, progress=crash_at_seventh)
    conn.execute("UPDATE analysis_runs SET status = 'running', finished_time = NULL")  # as left by power loss
    conn.commit()
    s = analyze(env)
    assert s.stale_runs_cancelled == 1
    assert (s.selection.done, s.processed) == (6, 6) and status(conn) == {"done": 12}


def test_failed_photo_is_retried_next_run(env, monkeypatch):
    _, _, conn = env
    real = diagnostic.load_image_with_orientation

    def flaky(path):
        if path.endswith("run_03.jpg"):
            raise RuntimeError("simulated read error")
        return real(path)
    monkeypatch.setattr(diagnostic, "load_image_with_orientation", flaky)
    first = analyze(env)
    assert (first.processed, first.failed_now) == (12, 1) and status(conn) == {"done": 11, "error": 1}

    monkeypatch.undo()
    second = analyze(env)
    assert (len(second.selection.failed), second.processed, second.failed_now) == (1, 1, 0)
    assert status(conn) == {"done": 12}


def test_moved_photo_keeps_results_and_unreadable_is_skipped(env):
    folder, _, conn = env
    analyze(env)
    before = rows(conn)
    (folder / "run_00.jpg").rename(folder / "renamed_00.jpg")
    (folder / "broken.jpg").write_bytes(b"\xff\xd8 not a photo")
    s = analyze(env)
    assert (s.scan.relinked, s.scan.unreadable, s.selection.unreadable, s.to_process) == (1, 1, 1, 0)
    assert [r[0] for r in rows(conn)] == [r[0] for r in before]


def test_other_settings_are_refused_unless_replace(env):
    folder, cfg, conn = env
    analyze(env)
    photo(folder / "extra.jpg")
    cfg.detection.crop_strategy = "torso"
    with pytest.raises(ResultConflict, match="different OCR settings"):
        analyze(env)
    s = analyze(env, replace=True)
    assert (len(s.selection.new), len(s.selection.other_settings), s.processed) == (1, 12, 13)
    assert analyze(env).to_process == 0


def test_device_change_alone_does_not_reanalyse(env):
    folder, cfg, _ = env
    analyze(env)
    cfg.ocr.device = "auto"
    photo(folder / "extra.jpg")
    s = analyze(env)                                   # no conflict: device is not a result setting
    assert (s.processed, s.selection.done) == (1, 12)


def test_only_photos_under_the_given_folder(env, tmp_path):
    other = tmp_path / "other_folder"
    photo(other / "x.jpg")
    _, cfg, conn = env
    analyze_folder(conn, other, "Race", cfg, Models())
    s = analyze(env)
    assert s.processed == 12 and s.selection.done == 0          # x.jpg is not counted here


def test_folder_inside_app_data_is_refused_not_silently_empty(env):
    _, cfg, conn = env
    inside = cfg.paths.data_dir / "photos"
    photo(inside / "a.jpg")
    with pytest.raises(ValueError, match="scanner always skips"):
        analyze_folder(conn, inside, "Race", cfg, Models())
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
