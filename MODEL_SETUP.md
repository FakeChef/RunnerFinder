# Model setup

RunnerFinder uses two pretrained models. Neither is stored in this repository.

| Model | Used for | Stored at | License (from the model files) |
|---|---|---|---|
| `yolo26n.pt` (Ultralytics YOLO26n, COCO-pretrained) | person detection | `models\yolo26n.pt` | the checkpoint states AGPL-3.0 |
| `PP-OCRv5_mobile_det`, `en_PP-OCRv5_mobile_rec` (PaddleOCR) | text detection and recognition | `models\paddlex\official_models\` | Apache-2.0 (model cards) |

## Download

With the engine environment installed (see [INSTALLATION.md](INSTALLATION.md)):

```
.\.venv\Scripts\python.exe check_environment.py --models
```

This downloads both models on first use and runs them once on a generated test image. That is a check that the models load and run; it does not measure accuracy or speed.

If the PaddleOCR download fails, select another model source and run the command again:

| Shell | Command |
|---|---|
| Command Prompt | `set PADDLE_PDX_MODEL_SOURCE=bos` |
| PowerShell | `$env:PADDLE_PDX_MODEL_SOURCE = "bos"` |

## Where the models are looked up

- YOLO: `models\yolo26n.pt` (`config.py` → `PathsConfig.models_dir`, `DetectionConfig.model_name`).
- PaddleOCR: `config.py` sets `PADDLE_PDX_CACHE_HOME` to the absolute path `models\paddlex` before PaddleOCR is imported, so the location is the same whatever folder a command is started from. An existing `PADDLE_PDX_CACHE_HOME` environment variable takes precedence.

## Offline use

- Once `models\yolo26n.pt` exists, `config.py` sets `YOLO_OFFLINE=1`, so Ultralytics makes no connectivity checks.
- `config.py` also sets `YOLO_AUTOINSTALL=False`, so Ultralytics never installs packages at runtime, and `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`, which skips PaddleOCR's model-server check on start-up.
- Ultralytics' usage analytics (`sync`) are switched off before the first prediction (`detection/person_detector.py`). This is stored in your Ultralytics user settings and also affects other Ultralytics programs for your Windows user. To undo it: `yolo settings sync=True`.
