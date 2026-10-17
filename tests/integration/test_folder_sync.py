"""Incremental scanning into a real SQLite database."""
import os
import shutil

import pytest

from config import ScanConfig
from database.database import get_or_create_event, init_db
from scanner.folder_sync import sync_folder


@pytest.fixture
def env(tmp_path, image_factory):
    conn = init_db(tmp_path / "db" / "test.db")
    photos = tmp_path / "photos"
    for i in range(1, 4):
        image_factory(photos / f"IMG_{i:04d}.jpg", datetime_original=f"2026:04:12 08:00:0{i}",
                      color=(i * 60, 30, 30))
    event = get_or_create_event(conn, "Race")
    yield conn, event, photos, image_factory
    conn.close()


def rows(conn, event):
    return {r["file_name"]: r for r in conn.execute(
        "SELECT * FROM photos WHERE event_id = ?", (event,))}


def test_first_scan_then_unchanged_rescan(env):
    conn, event, photos, _ = env
    s = sync_folder(conn, event, photos)
    assert (s.found, s.new, s.completed) == (3, 3, True)
    assert all(r["analysis_status"] == "pending" for r in rows(conn, event).values())
    s = sync_folder(conn, event, photos)
    assert (s.found, s.new, s.unchanged, s.changed) == (3, 0, 3, 0)


def test_changed_file_is_marked_for_reanalysis(env):
    conn, event, photos, make = env
    sync_folder(conn, event, photos)
    conn.execute("UPDATE photos SET analysis_status = 'done'")
    make(photos / "IMG_0002.jpg", size=(500, 300))          # re-edited / replaced
    s = sync_folder(conn, event, photos)
    assert s.changed == 1
    r = rows(conn, event)
    assert r["IMG_0002.jpg"]["analysis_status"] == "pending"
    assert r["IMG_0002.jpg"]["width"] == 500
    assert r["IMG_0001.jpg"]["analysis_status"] == "done"


def test_moved_and_renamed_file_keeps_its_id_and_corrections(env):
    conn, event, photos, _ = env
    sync_folder(conn, event, photos)
    old_id = rows(conn, event)["IMG_0001.jpg"]["id"]
    conn.execute("INSERT INTO detection_feedback (photo_id, verdict, corrected_bib) "
                 "VALUES (?, 'added', 'UT001')", (old_id,))
    (photos / "finish").mkdir()
    shutil.move(photos / "IMG_0001.jpg", photos / "finish" / "Runner_UT001.jpg")
    s = sync_folder(conn, event, photos)
    assert (s.relinked, s.new, s.newly_missing) == (1, 0, 0)
    r = rows(conn, event)["Runner_UT001.jpg"]
    assert r["id"] == old_id and r["file_status"] == "ok"
    assert conn.execute("SELECT bib_number FROM v_photo_bibs WHERE photo_id = ?",
                        (old_id,)).fetchone()[0] == "UT001"


def test_copied_file_is_a_new_photo_not_a_move(env):
    conn, event, photos, _ = env
    sync_folder(conn, event, photos)
    shutil.copy2(photos / "IMG_0001.jpg", photos / "copy.jpg")
    s = sync_folder(conn, event, photos)
    assert (s.new, s.relinked) == (1, 0)


def test_deleted_file_is_marked_missing_and_restored_later(env):
    conn, event, photos, _ = env
    sync_folder(conn, event, photos)
    backup = photos.parent / "backup.jpg"
    shutil.move(photos / "IMG_0003.jpg", backup)
    s = sync_folder(conn, event, photos)
    assert s.newly_missing == 1
    assert rows(conn, event)["IMG_0003.jpg"]["file_status"] == "missing"
    shutil.move(backup, photos / "IMG_0003.jpg")
    sync_folder(conn, event, photos)
    assert rows(conn, event)["IMG_0003.jpg"]["file_status"] == "ok"


def test_missing_root_changes_nothing(env):
    conn, event, photos, _ = env
    sync_folder(conn, event, photos)
    with pytest.raises(FileNotFoundError):
        sync_folder(conn, event, photos.parent / "unplugged_drive")
    assert all(r["file_status"] == "ok" for r in rows(conn, event).values())


def test_scanning_one_folder_does_not_touch_another(env, tmp_path):
    conn, event, photos, make = env
    cam_b = tmp_path / "camera_b"
    make(cam_b / "IMG_0001.jpg", color=(10, 200, 10))     # same name, other camera
    sync_folder(conn, event, photos)
    sync_folder(conn, event, cam_b)
    paths = [r["file_path"] for r in conn.execute("SELECT file_path FROM photos ORDER BY id")]
    assert len(paths) == 4 and sum(p.endswith("IMG_0001.jpg") for p in paths) == 2
    s = sync_folder(conn, event, cam_b)                    # rescan B only
    assert s.newly_missing == 0
    assert conn.execute("SELECT COUNT(*) FROM photos WHERE file_status = 'missing'").fetchone()[0] == 0


def test_interrupted_scan_resumes_and_marks_nothing_missing(env):
    conn, event, photos, make = env
    for i in range(4, 11):
        make(photos / f"IMG_{i:04d}.jpg", color=(i * 20, 10, 10))

    def stop_after_four(done, path):
        if done == 4:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        sync_folder(conn, event, photos, ScanConfig(commit_every=2), progress=stop_after_four)
    assert conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 4      # progress kept
    assert conn.execute("SELECT COUNT(*) FROM photos WHERE file_status='missing'").fetchone()[0] == 0
    s = sync_folder(conn, event, photos)
    assert (s.found, s.unchanged, s.new) == (10, 4, 6)


def test_unreadable_file_is_recorded_and_retried(env):
    conn, event, photos, make = env
    (photos / "broken.jpg").write_bytes(b"not an image")
    s = sync_folder(conn, event, photos)
    assert (s.found, s.new, s.unreadable) == (4, 3, 1)
    r = rows(conn, event)["broken.jpg"]
    assert r["file_status"] == "unreadable" and "UnidentifiedImageError" in r["file_error"]
    make(photos / "broken.jpg")                    # photographer replaces the file
    sync_folder(conn, event, photos)
    r2 = rows(conn, event)["broken.jpg"]
    assert r2["id"] == r["id"] and r2["file_status"] == "ok" and r2["width"] == 400


def test_same_folder_in_two_events_is_independent(env):
    conn, event, photos, _ = env
    other = get_or_create_event(conn, "Race (second index)")
    sync_folder(conn, event, photos)
    s = sync_folder(conn, other, photos)
    assert s.new == 3
    assert conn.execute("SELECT COUNT(DISTINCT event_id) FROM photos").fetchone()[0] == 2


def test_export_folder_inside_photo_folder_is_not_indexed(env, image_factory):
    conn, event, photos, _ = env
    image_factory(photos / "Export" / "Race" / "Bib_1" / "IMG_0001.jpg")
    s = sync_folder(conn, event, photos, exclude_dirs=(photos / "Export",))
    assert s.found == 3


def test_unplugged_other_folder_is_not_marked_missing(env, tmp_path):
    """Folder A's card is removed while folder B is scanned: A is left alone."""
    conn, event, photos, make = env
    cam_b = tmp_path / "camera_b"
    make(cam_b / "B_0001.jpg")
    sync_folder(conn, event, photos)
    sync_folder(conn, event, cam_b)
    shutil.move(photos, tmp_path / "card_removed")
    s = sync_folder(conn, event, cam_b)
    assert s.newly_missing == 0
    assert conn.execute("SELECT COUNT(*) FROM photos WHERE file_status = 'missing'").fetchone()[0] == 0
