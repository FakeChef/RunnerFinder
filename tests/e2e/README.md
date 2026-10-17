# End-to-end tests

These tests use the real AI packages instead of stand-ins. They are skipped automatically when the packages or models are missing.

| Test file | Checks | Needs |
|---|---|---|
| `test_ultralytics_pillow_patch.py` | the Ultralytics behaviour that `utils/pil_compat.py` works around still exists in the pinned version, and the scanner is unaffected | Ultralytics (no model) |
| `test_ultralytics_offline.py` | detection makes no network connections once the model is on disk | Ultralytics + `models/yolo26n.pt` |
| `test_yolo_detection.py` | real YOLO26n person detection on the sample photo shipped with Ultralytics | Ultralytics + `models/yolo26n.pt` |
| `test_paddleocr_reader.py` | PaddleOCR settings are accepted; real reading of synthetic bibs when the models are present | PaddleOCR (+ models in `models/paddlex`) |

Install the full engine environment and download the models first (see `MODEL_SETUP.md`). These tests check that the components work; they do not measure accuracy on real race photos.
