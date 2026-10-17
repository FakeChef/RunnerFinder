# Third-party components

This repository contains only RunnerFinder's own source code. It does **not** include third-party source code, binaries, or model weights. Dependencies are installed separately from PyPI with the requirement files, and the models are downloaded on first use (see [MODEL_SETUP.md](MODEL_SETUP.md)). Each component is licensed by its own authors under its own terms.

## Python packages (direct dependencies, pinned versions)

| Package | Version | License (per package metadata) | Used for |
|---|---|---|---|
| ultralytics | 8.4.165 | AGPL-3.0 | person detection (YOLO) |
| paddleocr | 3.7.0 | Apache-2.0 | bib OCR |
| paddlepaddle | 3.3.1 | Apache-2.0 | OCR runtime |
| torch / torchvision | 2.14.0 / 0.29.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT / BSD | detection runtime |
| opencv-python, opencv-contrib-python | 4.10.0.84 | Apache-2.0 (the wheels bundle FFmpeg under LGPL-2.1) | image I/O |
| PySide6-Essentials | 6.8.1 | LGPL-3.0 or GPL, or commercial (Qt) | desktop GUI |
| pillow | 12.3.0 | MIT-CMU | images, thumbnails |
| numpy | 2.3.5 | BSD-3-Clause | arrays |
| pyyaml | 6.0.2 | MIT | rule files |
| streamlit | 1.64.0 | Apache-2.0 | legacy web UI |
| pytest | 9.1.1 | MIT | tests |

The hash-locked `requirements-*-lock-win-py311.txt` files list every transitive package.

## Models (downloaded separately, not in this repository)

| Model | License | Source |
|---|---|---|
| YOLO26n (`yolo26n.pt`) | AGPL-3.0 (stated in the checkpoint and by Ultralytics) | Ultralytics |
| PP-OCRv5 mobile detection and English recognition | Apache-2.0 (model cards) | PaddlePaddle / PaddleOCR |

## GPU libraries

The optional GPU setup uses `paddlepaddle-gpu` and NVIDIA CUDA/cuDNN libraries, which are distributed by NVIDIA under NVIDIA's proprietary license terms. They are not part of this repository.

## Windows installer

Binary installers that bundle these components (Python runtime, packages, Qt, NVIDIA libraries and models) carry additional obligations, such as license texts, source offers and NVIDIA's redistribution terms. Those obligations are **not** covered by this note and are still being reviewed.
