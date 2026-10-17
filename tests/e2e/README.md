# End-to-end tests (planned, Phases 4-8)

These will run the complete pipeline (scan -> YOLO -> PaddleOCR -> SQLite
-> search -> export) on a few test images. They need the AI stack and the
downloaded models, and are skipped automatically when those are missing.

They check that the pipeline WORKS end to end. They do not measure
accuracy on real race photos; that is done by `evaluation/evaluate.py`
with your labelled photos.
