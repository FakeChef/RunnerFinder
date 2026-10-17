-- ======================================================================
-- Runner Photo Finder - SQLite schema, version 2
-- ======================================================================
--
-- COORDINATE SYSTEM
--   Every pixel coordinate in this database (bbox_*, person_bbox_*) and
--   photos.width / photos.height refer to the ORIGINAL full-resolution
--   image AFTER applying its EXIF orientation, i.e. the photo as it looks
--   in a normal viewer. Origin (0,0) is the top-left corner.
--   Detection may run on a downscaled copy; results are scaled back to
--   this system before they are stored.
--
-- BIB NUMBERS are TEXT everywhere, never integers, so leading zeros are
--   preserved ("UT001", "00123"). Always pass them as Python str.
--
-- ORIGINAL PHOTOS are never modified. The database only stores paths
--   and metadata; files are opened read-only.
-- ======================================================================

-- One row per race. Every photo belongs to exactly one event.
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE CHECK (trim(name) <> ''),
    event_date      TEXT,
    location        TEXT,
    notes           TEXT,
    created_time    TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- Optional list of registered bibs per event (from the organiser).
-- When present, OCR results not on the list can be rejected.
CREATE TABLE IF NOT EXISTS event_bibs (
    event_id        INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    bib_number      TEXT    NOT NULL,
    runner_name     TEXT,
    category        TEXT,
    PRIMARY KEY (event_id, bib_number)
);

-- One row per AI analysis run: which models/settings produced which results.
CREATE TABLE IF NOT EXISTS analysis_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id            INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    source_folder       TEXT    NOT NULL,
    status              TEXT    NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'finished', 'cancelled', 'failed')),
    detector_model      TEXT,
    ocr_model           TEXT,
    settings_json       TEXT,
    photos_total        INTEGER,
    photos_processed    INTEGER DEFAULT 0,
    started_time        TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    finished_time       TEXT
);

CREATE TABLE IF NOT EXISTS photos (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id            INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,

    -- File identity. path_key is the normalised path (case-folded on
    -- Windows) used for uniqueness, so "D:\A\IMG_1.JPG" and
    -- "d:\a\img_1.jpg" are the same file. Two files with the same NAME in
    -- different folders have different paths and never collide.
    file_path           TEXT    NOT NULL,           -- absolute, as found on disk
    path_key            TEXT    NOT NULL,
    file_name           TEXT    NOT NULL,
    file_size           INTEGER,
    file_mtime_ns       INTEGER,                    -- size+mtime detect changes
    fingerprint         TEXT,                       -- recognises moved files

    -- 'ok' | 'missing' (not found at last scan; row and corrections kept)
    -- | 'unreadable' (exists but cannot be opened as an image)
    file_status         TEXT    NOT NULL DEFAULT 'ok'
                        CHECK (file_status IN ('ok', 'missing', 'unreadable')),
    file_error          TEXT,

    -- Image metadata. width/height are EXIF-oriented (as displayed).
    width               INTEGER,
    height              INTEGER,
    exif_orientation    INTEGER,                    -- 1-8, NULL if absent
    capture_time        TEXT,     -- ISO 8601 local time, e.g. 2026-04-12T08:31:22.450
    capture_offset      TEXT,     -- e.g. +08:00 when the camera recorded it
    capture_time_source TEXT,     -- exif_original | exif_digitized | exif_modified
    camera_make         TEXT,
    camera_model        TEXT,
    camera_serial       TEXT,

    -- AI analysis state
    analysis_status     TEXT    NOT NULL DEFAULT 'pending'
                        CHECK (analysis_status IN ('pending', 'done', 'error')),
    analysis_error      TEXT,
    processing_ms       INTEGER,
    run_id              INTEGER REFERENCES analysis_runs(id) ON DELETE SET NULL,

    created_time        TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    last_seen_time      TEXT,
    UNIQUE (event_id, path_key)
);

-- One row per region the pipeline examined in a photo. Several rows per
-- photo are normal (group photos). bib_number is NULL when a region was
-- found but no valid bib could be read; those rows let detection and OCR
-- accuracy be measured separately.
CREATE TABLE IF NOT EXISTS bib_detection (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id            INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    run_id              INTEGER REFERENCES analysis_runs(id) ON DELETE SET NULL,
    bib_number          TEXT    CHECK (bib_number IS NULL OR bib_number <> ''),
    confidence          REAL,                       -- combined, used for ranking
    det_confidence      REAL,                       -- stage 1 score
    ocr_confidence      REAL,                       -- stage 2 score
    bbox_x              INTEGER CHECK (bbox_x >= 0),   -- bib text box
    bbox_y              INTEGER CHECK (bbox_y >= 0),
    bbox_width          INTEGER CHECK (bbox_width >= 0),
    bbox_height         INTEGER CHECK (bbox_height >= 0),
    person_bbox_x       INTEGER,                    -- runner box, if any
    person_bbox_y       INTEGER,
    person_bbox_width   INTEGER,
    person_bbox_height  INTEGER,
    ocr_text            TEXT,                       -- raw OCR text before cleaning
    source              TEXT    NOT NULL
                        CHECK (source IN ('person_body', 'full_image', 'bib_model')),
    created_time        TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- Manual verification by the photographer.
--
-- Feedback is attached to (photo, bib number), NOT to a detection row.
-- Re-analysis deletes and recreates bib_detection rows, but feedback rows
-- are never deleted by the pipeline and keep applying to the new results:
--   correct    original_bib confirmed           -> always shown, confidence 1.0
--   incorrect  original_bib is wrong / not a bib -> hidden, even if re-detected
--   corrected  original_bib replaced by corrected_bib
--   added      AI missed this bib; corrected_bib added by hand
-- detection_id and the bbox/ocr_text snapshot are kept for future model
-- training; detection_id becomes NULL when that detection row is replaced.
CREATE TABLE IF NOT EXISTS detection_feedback (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id            INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    detection_id        INTEGER REFERENCES bib_detection(id) ON DELETE SET NULL,
    verdict             TEXT    NOT NULL
                        CHECK (verdict IN ('correct', 'incorrect', 'corrected', 'added')),
    original_bib        TEXT,
    corrected_bib       TEXT,
    ocr_text            TEXT,
    bbox_x              INTEGER,
    bbox_y              INTEGER,
    bbox_width          INTEGER,
    bbox_height         INTEGER,
    note                TEXT,
    created_time        TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_time        TEXT,
    CHECK (
        (verdict IN ('correct', 'incorrect') AND original_bib IS NOT NULL) OR
        (verdict = 'corrected' AND original_bib IS NOT NULL
                               AND corrected_bib IS NOT NULL AND corrected_bib <> '') OR
        (verdict = 'added'     AND original_bib IS NULL
                               AND corrected_bib IS NOT NULL AND corrected_bib <> '')
    )
);

-- One verdict per AI reading per photo; one manual "added" row per bib per photo.
CREATE UNIQUE INDEX IF NOT EXISTS ux_feedback_original
    ON detection_feedback(photo_id, original_bib) WHERE original_bib IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_feedback_added
    ON detection_feedback(photo_id, corrected_bib) WHERE verdict = 'added';

CREATE INDEX IF NOT EXISTS ix_detection_bib        ON bib_detection(bib_number);
CREATE INDEX IF NOT EXISTS ix_detection_photo      ON bib_detection(photo_id);
CREATE INDEX IF NOT EXISTS ix_feedback_corrected   ON detection_feedback(corrected_bib);
CREATE INDEX IF NOT EXISTS ix_photos_event_time    ON photos(event_id, capture_time);
CREATE INDEX IF NOT EXISTS ix_photos_fingerprint   ON photos(event_id, fingerprint);
CREATE INDEX IF NOT EXISTS ix_photos_analysis      ON photos(event_id, analysis_status);

-- Effective bibs per photo, used by search: AI results with the
-- photographer's feedback applied, one row per (photo, bib).
CREATE VIEW IF NOT EXISTS v_photo_bibs AS
SELECT photo_id, bib_number, MAX(confidence) AS confidence, verification
FROM (
    -- Confirmed, corrected or added by the photographer
    SELECT f.photo_id,
           CASE WHEN f.verdict = 'correct' THEN f.original_bib
                ELSE f.corrected_bib END AS bib_number,
           1.0       AS confidence,
           f.verdict AS verification
    FROM detection_feedback f
    WHERE f.verdict IN ('correct', 'corrected', 'added')

    UNION ALL

    -- AI results without any feedback on that (photo, bib)
    SELECT d.photo_id, d.bib_number, d.confidence, 'unverified'
    FROM bib_detection d
    WHERE d.bib_number IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM detection_feedback f
                      WHERE f.photo_id = d.photo_id
                        AND f.original_bib = d.bib_number)
)
-- SQLite returns "verification" from the row with the highest confidence,
-- so manual feedback (1.0) wins over a duplicate AI reading.
GROUP BY photo_id, bib_number;
