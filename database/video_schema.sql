-- ======================================================================
-- RunnerFinder V1.3 - video extension (extension "video", version 1)
-- ======================================================================
-- ADDITIVE ONLY: no existing table, index or view is changed, and
-- PRAGMA user_version stays 3, so RunnerFinder 1.2 can still open the
-- file (it ignores these tables). Applied by database/video_store.py in one
-- transaction after a backup. Original videos are never modified: only
-- paths, metadata and recognition results are stored (no video data).
-- Bib numbers are TEXT (leading zeros kept), as everywhere.
-- ======================================================================

CREATE TABLE IF NOT EXISTS schema_extensions (
    name            TEXT    PRIMARY KEY,
    version         INTEGER NOT NULL,
    applied_time    TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- One row per video file of an event (same identity rules as photos).
CREATE TABLE IF NOT EXISTS videos (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id            INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    file_path           TEXT    NOT NULL,
    path_key            TEXT    NOT NULL,
    file_name           TEXT    NOT NULL,
    file_size           INTEGER,
    file_mtime_ns       INTEGER,
    fingerprint         TEXT,
    file_status         TEXT    NOT NULL DEFAULT 'ok'
                        CHECK (file_status IN ('ok', 'missing', 'unreadable')),
    file_error          TEXT,
    duration_s          REAL,
    fps                 REAL,
    frame_count         INTEGER,
    width               INTEGER,          -- as displayed (rotation applied)
    height              INTEGER,
    analysis_status     TEXT    NOT NULL DEFAULT 'pending'
                        CHECK (analysis_status IN ('pending', 'done', 'error')),
    analysis_error      TEXT,
    -- recognition result (NULL until analysed)
    recognition_status  TEXT    CHECK (recognition_status IS NULL OR
                                       recognition_status IN ('VERIFIED', 'AMBIGUOUS', 'UNINDEXED')),
    strategy            TEXT,             -- e.g. 'adaptive_s5_s7/1'
    frames_used         INTEGER,          -- 5 or 7
    escalated           INTEGER,          -- 1 if the two S7 frames were added
    sample_times_json   TEXT,             -- presentation times (s) of the sampled frames
    evidence_json       TEXT,             -- per-frame readings, tallies, decision
    thumbnail_fingerprint TEXT,           -- key in the thumbnail cache (representative frame)
    processing_ms       INTEGER,
    run_id              INTEGER REFERENCES analysis_runs(id) ON DELETE SET NULL,
    analysis_settings_hash TEXT,
    created_time        TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    last_seen_time      TEXT,
    UNIQUE (event_id, path_key)
);

-- Vote candidates of a video. role 'accepted': the one verified bib of a VERIFIED video.
-- role 'candidate': every other key with votes (AMBIGUOUS evidence; never a confirmed match).
CREATE TABLE IF NOT EXISTS video_candidates (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id            INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    bib_number          TEXT    NOT NULL CHECK (bib_number <> ''),   -- best full form read
    numeric_bib         TEXT    NOT NULL CHECK (numeric_bib <> ''),  -- the vote key
    role                TEXT    NOT NULL CHECK (role IN ('accepted', 'candidate')),
    votes               INTEGER NOT NULL,
    frames              INTEGER NOT NULL,     -- frames evaluated (5 or 7)
    best_confidence     REAL,
    forms               TEXT,                 -- all full forms read, e.g. "A15293|15293"
    UNIQUE (video_id, numeric_bib)
);

-- Photographer review of a video bib (same meaning as detection_feedback 'incorrect').
CREATE TABLE IF NOT EXISTS video_feedback (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id            INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    numeric_bib         TEXT    NOT NULL,
    verdict             TEXT    NOT NULL CHECK (verdict IN ('incorrect')),
    created_time        TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    UNIQUE (video_id, numeric_bib)
);

CREATE INDEX IF NOT EXISTS ix_videos_event          ON videos(event_id, analysis_status);
CREATE INDEX IF NOT EXISTS ix_videos_fingerprint    ON videos(event_id, fingerprint);
CREATE INDEX IF NOT EXISTS ix_video_candidates_num  ON video_candidates(numeric_bib, role);
