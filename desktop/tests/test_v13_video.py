"""V1.3 video integration tests that need no AI models (standard library + sqlite).

    <desktop or engine python> -m unittest desktop.tests.test_v13_video     (cwd: the source folder)

* video/recognize.py: Adaptive S5->S7 rules (thresholds, escalation, conflict, ambiguity, one vote per
  frame, noise filter, year safeguard, leading zeros)
* B2 reproduction: the decision code replayed on the recorded B1 frame evidence of the 30 clips gives
  the B2 statuses and bibs (ADAPT, noise-filtered)
* database/video_store.py: additive migration (backup, preservation, user_version 3, idempotent,
  rollback on failure)
* search/video_search.py: exact / numeric tiers, no partial matches, leading zeros, ambiguous and
  rejected handling, event filter, index without video tables
* RunnerFinder 1.2 (RUNNERFINDER_V12_SOURCE) opens and searches an extended index unchanged
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from database import database as db
from database import video_store as vs
from search import video_search
from video import recognize as rz
from desktop.tests.private_data import V12_INDEX, V12_SOURCE, VIDEO_EVIDENCE, available

B1 = VIDEO_EVIDENCE
V02 = V12_SOURCE
PROD_BACKUP = V12_INDEX
YEARS = ("2024", "2025", "2026")


def fb(bib, conf=0.99, ok=True, numeric=None):
    num = numeric if numeric is not None else ("".join(c for c in bib if c.isdigit()) or None)
    return rz.FrameBib(bib, num, conf, ok)


class DecisionRulesTest(unittest.TestCase):
    def test_positions_and_thresholds_are_the_b2_values(self):
        self.assertEqual(rz.S5_POSITIONS, (0.1, 0.3, 0.5, 0.7, 0.9))
        self.assertEqual(rz.ESCALATION_POSITIONS, (0.214286, 0.785714))
        self.assertEqual((rz.S5_THRESHOLD, rz.S7_THRESHOLD, rz.ESCALATE_MIN_VOTES), (3, 4, 2))

    def test_s5_verified_without_escalation(self):
        frames = [[fb("A15293")], [fb("15293")], [], [fb("15293")], [fb("1529", ok=False)]]
        d = rz.decide_adaptive(frames, lambda: self.fail("must not escalate"), YEARS)
        self.assertEqual((d.status, d.key, d.escalated, d.frames_used), ("VERIFIED", "15293", False, 5))
        self.assertEqual(d.full_bib(), "A15293")

    def test_escalation_to_s7(self):
        frames = [[fb("10460")], [], [fb("10460")], [], []]
        d = rz.decide_adaptive(frames, lambda: [[fb("10460")], [fb("10460")]], YEARS)
        self.assertEqual((d.status, d.key, d.escalated, d.frames_used), ("VERIFIED", "10460", True, 7))

    def test_escalation_not_verified_at_3_of_7(self):
        frames = [[fb("16966")], [], [fb("16966")], [], []]
        d = rz.decide_adaptive(frames, lambda: [[fb("16966")], []], YEARS)
        self.assertEqual((d.status, d.key, d.escalated), ("UNINDEXED", None, True))

    def test_no_escalation_when_best_has_one_vote(self):
        frames = [[fb("90101", 0.61)], [], [], [], []]
        d = rz.decide_adaptive(frames, lambda: self.fail("must not escalate"), YEARS)
        self.assertEqual((d.status, d.escalated), ("UNINDEXED", False))

    def test_two_runners_are_ambiguous_never_verified(self):
        frames = [[fb("15293")], [fb("15293")], [fb("19791")], [fb("19791")], []]
        d = rz.decide_adaptive(frames, lambda: [[fb("19791")], [fb("15293")]], YEARS)
        self.assertEqual((d.status, d.key), ("AMBIGUOUS", None))

    def test_conflict_is_ambiguous(self):
        frames = [[fb("11111"), fb("22222")]] * 3 + [[], []]
        t = rz.tally(frames, YEARS)
        self.assertEqual(rz.decide(t, 3), ("AMBIGUOUS", None))

    def test_one_vote_per_frame_per_key(self):
        frames = [[fb("39151"), fb("B39151", numeric="39151"), fb("39151", 0.5)]] + [[]] * 4
        self.assertEqual(rz.tally(frames, YEARS).votes, {"39151": 1})

    def test_noise_filtered_bibs_do_not_vote(self):
        frames = [[fb("13116998", ok=False)]] * 5
        self.assertEqual(rz.tally(frames, YEARS).votes, {})

    def test_year_keys_never_vote(self):
        frames = [[fb("2026")], [fb("42026")], [fb("02026")], [fb("62026")], [fb("67933")]]
        t = rz.tally(frames, YEARS)
        self.assertEqual(t.votes, {"67933": 1})

    def test_leading_zeros_are_kept_in_keys(self):
        frames = [[fb("01234")]] * 3 + [[], []]
        d = rz.decide_adaptive(frames, lambda: [], YEARS)
        self.assertEqual(d.key, "01234")


class B2ReproductionTest(unittest.TestCase):
    """The integrated decision code on the B1-recorded frame evidence must give the B2 results."""

    def test_30_clips_reproduce_b2(self):
        if not available(B1):
            self.skipTest("RUNNERFINDER_VIDEO_EVIDENCE not set or not found")
        ev_path, b2_path = B1 / "data" / "ai_evidence.jsonl", B1 / "data" / "b2_per_video.csv"
        if not ev_path.exists() or not b2_path.exists():
            self.skipTest("B1/B2 evidence not available")
        b2 = {r["file_name"]: r for r in csv.DictReader(open(b2_path, encoding="utf-8-sig"))
              if r["strategy"] == "ADAPT" and r["candidates"] == "filtered"}
        n = 0
        for line in open(ev_path, encoding="utf-8"):
            r = json.loads(line)
            frames = {round(f["fraction"], 6): [rz.FrameBib(b["bib"], b.get("numeric"), b["conf"], b["filter_accepted"])
                                                 for b in f.get("bibs", [])] for f in r["frames"]}
            d = rz.decide_adaptive([frames[f] for f in rz.S5_POSITIONS],
                                   lambda: [frames[f] for f in rz.ESCALATION_POSITIONS], YEARS)
            ref = b2[r["file_name"]]
            want_bib = ref["bib"] if ref["status"] == "VERIFIED" else None
            self.assertEqual(d.key, want_bib, r["file_name"])
            if ref["status"] == "VERIFIED":
                self.assertEqual(d.status, "VERIFIED", r["file_name"])
            else:
                self.assertIn(d.status, ("AMBIGUOUS", "UNINDEXED"), r["file_name"])
            if ref["label_kind"] == "MULTI":
                self.assertEqual(d.status, "AMBIGUOUS", r["file_name"])
            n += 1
        self.assertEqual(n, 30)


def _make_v3(path: Path) -> None:
    """A schema-3 index created by the unchanged V1.2 schema file, with one event, photo and bib."""
    conn = sqlite3.connect(path)
    conn.executescript(db.SCHEMA_FILE.read_text(encoding="utf-8"))      # the 1.2 schema: no video tables
    conn.execute("PRAGMA user_version = 3")
    conn.execute("INSERT INTO events (name) VALUES ('race')")
    conn.execute("INSERT INTO photos (event_id, file_path, path_key, file_name) VALUES (1, 'D:\\p\\a.jpg', 'd:\\p\\a.jpg', 'a.jpg')")
    conn.execute("INSERT INTO bib_detection (photo_id, bib_number, numeric_bib, source) VALUES (1, 'B60914', '60914', 'person_body')")
    conn.execute("INSERT INTO detection_feedback (photo_id, verdict, original_bib, numeric_bib) VALUES (1, 'incorrect', 'B60914', '60914')")
    conn.commit()
    conn.close()


def _dump(path: Path, tables=("events", "photos", "bib_detection", "detection_feedback", "analysis_runs",
                              "event_bibs", "imported_batches")) -> dict:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return {t: conn.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in tables}
    finally:
        conn.close()


class MigrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rf13_mig_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_new_index_gets_the_extension_without_backup(self):
        conn = db.init_db(self.tmp / "new.db")
        self.assertEqual(vs.extension_version(conn), 1)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 3)
        conn.close()
        self.assertFalse((self.tmp / "backups").exists())

    def test_existing_index_is_backed_up_and_preserved(self):
        p = self.tmp / "v12.db"
        _make_v3(p)
        before = _dump(p)
        objects_before = sqlite3.connect(p).execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY name").fetchall()
        conn = db.init_db(p)
        self.assertEqual(vs.extension_version(conn), 1)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 3)
        self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        conn.close()
        self.assertEqual(_dump(p), before)
        after = {(t, n): s for t, n, s in sqlite3.connect(p).execute("SELECT type, name, sql FROM sqlite_master")}
        for t, n, s in objects_before:                          # every old object unchanged
            self.assertEqual(after[(t, n)], s, n)
        self.assertEqual({n for _, n in after} - {n for _, n, _ in objects_before},
                         {"schema_extensions", "videos", "video_candidates", "video_feedback",
                          "ix_videos_event", "ix_videos_fingerprint", "ix_video_candidates_num",
                          "sqlite_autoindex_schema_extensions_1", "sqlite_autoindex_videos_1",
                          "sqlite_autoindex_video_candidates_1", "sqlite_autoindex_video_feedback_1"})
        backups = list((self.tmp / "backups").glob("v12.pre_video.*.db"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(_dump(backups[0]), before)
        self.assertEqual(vs.extension_version(sqlite3.connect(backups[0])), 0)

    def test_idempotent(self):
        p = self.tmp / "v12.db"
        _make_v3(p)
        db.init_db(p).close()
        db.init_db(p).close()
        self.assertEqual(len(list((self.tmp / "backups").glob("*.db"))), 1)

    def test_failure_rolls_back_completely(self):
        p = self.tmp / "v12.db"
        _make_v3(p)
        before = _dump(p)
        bad = self.tmp / "bad.sql"
        bad.write_text("CREATE TABLE videos (id INTEGER PRIMARY KEY);\nCREATE TABLE broken (;\n", encoding="utf-8")
        with mock.patch.object(vs, "VIDEO_SCHEMA_FILE", bad):
            with self.assertRaises(Exception):
                db.init_db(p)
        conn = sqlite3.connect(p)
        self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'videos'").fetchone())
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 3)
        conn.close()
        self.assertEqual(_dump(p), before)
        conn = db.init_db(p)                                        # a later start succeeds
        self.assertEqual(vs.extension_version(conn), 1)
        conn.close()


def _add_video(conn, event_id, name, status, cands, path=None):
    vid = vs.insert_video(conn, event_id, {"file_path": path or f"Z:\\no\\{name}", "path_key": f"z:\\no\\{name.lower()}",
                                          "file_name": name, "fingerprint": "ab" * 32})
    vs.save_video_result(conn, vid, None, {"status": status, "strategy": rz.STRATEGY, "frames_used": 5,
                                           "escalated": False, "sample_times": [], "evidence": {},
                                           "candidates": cands})
    return vid


def cand(bib, numeric, role, votes):
    return {"bib_number": bib, "numeric_bib": numeric, "role": role, "votes": votes, "frames": 5}


class VideoSearchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rf13_vs_"))
        self.conn = db.init_db(self.tmp / "i.db")
        self.e1 = db.get_or_create_event(self.conn, "race")
        self.e2 = db.get_or_create_event(self.conn, "other")
        self.real = self.tmp / "real.mp4"
        self.real.write_bytes(b"x")
        self.v_ok = _add_video(self.conn, self.e1, "a.mp4", "VERIFIED",
                               [cand("A60914", "60914", "accepted", 3), cand("2", "2", "candidate", 1)], str(self.real))
        self.v_zero = _add_video(self.conn, self.e1, "b.mp4", "VERIFIED", [cand("01234", "01234", "accepted", 4)])
        self.v_amb = _add_video(self.conn, self.e1, "c.mp4", "AMBIGUOUS",
                                [cand("15293", "15293", "candidate", 2), cand("19791", "19791", "candidate", 3),
                                 cand("11111", "11111", "candidate", 1)])
        self.v_other = _add_video(self.conn, self.e2, "d.mp4", "VERIFIED", [cand("60914", "60914", "accepted", 3)])

    def tearDown(self):
        self.conn.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def s(self, q, event=None, **kw):
        return video_search.search_videos(self.conn, q, event, **kw)

    def test_numeric_and_exact_tiers(self):
        r = self.s("60914", self.e1)
        self.assertEqual([(m.video_id, m.match, m.bib, m.confirmed, m.missing) for m in r],
                         [(self.v_ok, "numeric", "A60914", True, False)])
        r = self.s("60914")
        self.assertEqual({(m.video_id, m.match) for m in r}, {(self.v_ok, "numeric"), (self.v_other, "exact")})

    def test_no_partial_or_fuzzy_matches_and_leading_zeros(self):
        for q in ("6091", "609140", "0914", "60915", "234", "6O914", ""):
            self.assertEqual(self.s(q), [], q)
        self.assertEqual([(m.video_id, m.match) for m in self.s("01234")], [(self.v_zero, "exact")])
        # V1.3.1: leading zeros only -> a labelled alias, never an exact or numeric match
        for q in ("1234", "001234"):
            self.assertEqual([(m.video_id, m.match) for m in self.s(q)], [(self.v_zero, "alias")], q)

    def test_candidates_of_verified_videos_are_never_matches(self):
        self.assertEqual(self.s("2"), [])
        self.assertEqual(self.s("2", include_ambiguous=True), [])

    def test_ambiguous_hidden_by_default_and_unconfirmed_when_shown(self):
        self.assertEqual(self.s("19791"), [])
        self.assertEqual(self.s("15293"), [])
        r = self.s("19791", include_ambiguous=True)
        self.assertEqual([(m.video_id, m.confirmed, m.recognition_status) for m in r],
                         [(self.v_amb, False, "AMBIGUOUS")])
        self.assertEqual(self.s("11111", include_ambiguous=True), [])      # 1 vote: not even a candidate

    def test_reject_and_restore(self):
        vs.reject_video_bib(self.conn, self.v_ok, "60914")
        self.assertEqual(self.s("60914", self.e1), [])
        r = self.s("60914", self.e1, include_rejected=True)
        self.assertTrue(r and r[0].rejected)
        vs.reject_video_bib(self.conn, self.v_ok, "60914")                 # idempotent
        vs.restore_video_bib(self.conn, self.v_ok, "60914")
        self.assertEqual(len(self.s("60914", self.e1)), 1)

    def test_missing_original_is_reported(self):
        self.assertTrue(self.s("01234")[0].missing)

    def test_counts(self):
        c = video_search.video_counts(self.conn)
        self.assertEqual((c[self.e1]["videos"], c[self.e1]["verified"], c[self.e1]["ambiguous"]), (3, 2, 1))

    def test_index_without_video_tables(self):
        conn = sqlite3.connect(self.tmp / "plain.db")
        conn.execute("CREATE TABLE x (a)")
        self.assertEqual(video_search.search_videos(conn, "60914"), [])
        self.assertEqual(video_search.video_counts(conn), {})
        conn.close()

    def test_reanalysis_replaces_candidates_without_duplicates(self):
        vs.save_video_result(self.conn, self.v_ok, None, {"status": "VERIFIED", "strategy": rz.STRATEGY,
                                                          "frames_used": 5, "escalated": False, "sample_times": [],
                                                          "evidence": {}, "candidates": [cand("A60914", "60914", "accepted", 4)]})
        n = self.conn.execute("SELECT COUNT(*) FROM video_candidates WHERE video_id = ?", (self.v_ok,)).fetchone()[0]
        self.assertEqual(n, 1)


class V12CompatibilityTest(unittest.TestCase):
    """RunnerFinder 1.2 (unchanged v02 code) must open, upgrade-check and search an extended index."""

    def test_v12_opens_extended_copy_of_production_backup(self):
        if not (available(PROD_BACKUP) and available(V02)):
            self.skipTest("RUNNERFINDER_V12_INDEX / RUNNERFINDER_V12_SOURCE not set or not found")
        tmp = Path(tempfile.mkdtemp(prefix="rf13_v12_"))
        try:
            p = tmp / "copy.db"
            src = sqlite3.connect(f"file:{PROD_BACKUP.as_posix()}?mode=ro", uri=True)
            dst = sqlite3.connect(p)
            src.backup(dst)
            src.close()
            dst.close()
            db.init_db(p).close()                                   # V1.3 adds the video tables
            code = ("import sys; sys.path.insert(0, r'%s'); sys.dont_write_bytecode = True\n"
                    "from database.database import init_db\nfrom search.db_search import search\n"
                    "c = init_db(r'%s'); g, f = search(c, '60914'); "
                    "print(c.execute('PRAGMA user_version').fetchone()[0], c.execute('PRAGMA integrity_check').fetchone()[0])"
                    % (V02, p))
            r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd=str(tmp))
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertEqual(r.stdout.split(), ["3", "ok"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
