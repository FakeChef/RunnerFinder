# RunnerFinder

[English](README.md) | **简体中文**

在你自己的 Windows 电脑上，按号码布（例如 `10258`、`T1234`、`UT001`）查找比赛照片和视频片段。

RunnerFinder 扫描一个赛事的照片和视频文件夹，用人体检测模型找出画面中的跑者，用 OCR 识别号码布，并把结果存入本地 SQLite 索引，之后即可按号码搜索。照片不会上传，原始文件只读取、从不修改。

**适合谁使用：** 需要在大量赛事照片中快速找出某位跑者的摄影师和赛事志愿者，以及想参与开发的开发者。

版本：**1.3.1**（见 `desktop/__init__.py`）

> 本 README 根据英文 [README.md](README.md) 和当前源代码编写。其余文档（[INSTALLATION.md](INSTALLATION.md)、[MODEL_SETUP.md](MODEL_SETUP.md)、[DEVELOPMENT.md](DEVELOPMENT.md)、[TESTING.md](TESTING.md)）目前只有英文版。

## 仓库包含什么

| 部分 | 启动方式 | 功能 |
|---|---|---|
| 桌面应用（PySide6） | `pythonw desktop\RunnerFinder.pyw` 或 `python -m desktop` | 仪表盘（Dashboard）、新建赛事并分析文件夹（New Event）、分析进度（Analysis Progress）、搜索（Search） |
| 命令行 | `python index_cli.py <命令>` | `scan`、`analyze`、`thumbs`、`filter-bibs`、`import-csv`、`detect`、`ocr`、`benchmark`、`gpu-check` |
| 旧版网页界面（Streamlit） | `run_app.bat` | 早期的搜索界面，只能在本机访问（`localhost`） |

支持的文件：照片 `.jpg`、`.jpeg`、`.png`；视频 `.mp4`、`.mov`、`.m4v`（扩展名不区分大小写）。

## 工作原理

### 照片索引
- **扫描**：读取文件夹及其子文件夹中的照片和 EXIF 信息（`scanner/`）。原始文件只以只读方式打开。
- **增量与断点续跑**：`index_cli.py analyze` 只处理新增、内容有变化或上次失败的照片。中断后再次运行同一命令会从中断处继续，不会产生重复结果。移动或改名的照片会通过内容指纹保留原有结果。这些行为由集成测试（`tests/integration/test_analyze_incremental.py`，使用替身 OCR 引擎）验证。
- **缩略图**：分析时为搜索页生成缩略图，保存在 `data\thumbnails\`。

### 号码布识别
1. **人体检测**：Ultralytics **YOLO26n** 找出画面中的人。
2. **OCR**：**PaddleOCR**（PP-OCRv5 mobile 模型：`PP-OCRv5_mobile_det`、`en_PP-OCRv5_mobile_rec`）读取人体区域内的文字。`analyze` 默认使用 `lite1` 模式，即只对排名最靠前的一个人体区域做 OCR。
3. **噪声过滤**：基于规则的过滤器（`config/bib_rules.yaml`）把不像号码布的数字（如年份、距离）标记为噪声并在默认结果中隐藏，**不会删除任何数据**。

### 按号码搜索
- 号码始终按**文本**处理，前导零会保留（`UT001`、`00123`）。OCR 结果、搜索输入和人工修正都按同一规则规范化：转为大写，去掉空白和分隔符 `- _ . # · : /`（`ocr/bib_format.py`）。长度和格式规则在 `config.py` → `BibFormatConfig` 中。
- 匹配分级（`search/db_search.py`），从高到低：
  1. **精确**：完整号码相同；
  2. **数字**：只有数字部分相同（例如搜 `0175` 能找到 `A0175`）；
  3. **规范化**：OCR 原文去掉分隔符后相同；
  4. **别名**：只差前导零（`175` ↔ `0175`）。
- 不做模糊匹配或部分匹配。用带字母的完整号码搜索时（如 `A0175`），不会返回其他字母前缀的结果（如 `B0175`）。
- 在桌面应用中可以把识别错误的结果标记为错误，也可以恢复。这只记录人工反馈，不改动 AI 识别结果。

## 视频支持（1.3 起）

**已实现，并有自动测试：**
- 在桌面应用中分析赛事文件夹时，会同时处理其中的视频，默认开启。
- 每个视频抽取固定位置的帧，用与照片相同的 OCR 和噪声过滤识别号码，再投票决定结果（`video/recognize.py`）：
  - 先取时长的 10 %、30 %、50 %、70 %、90 % 五帧，某个号码得到至少 3 票即判定为 **VERIFIED**；
  - 如果没有判定，而最多票的号码有 2 票，再加两帧（21.4 %、78.6 %），七帧中得到至少 4 票即为 VERIFIED；
  - 否则视频不会被索引为确认匹配，而是标记为 **AMBIGUOUS**（多个号码各得至少 2 票，或票数冲突）或 **UNINDEXED**；
  - 包含规则文件中年份的号码（如 `2026`）不参与投票。
- 搜索结果中，视频标为 `VIDEO`，照片标为 `PHOTO`。AMBIGUOUS 视频的候选号码只在勾选相应选项后显示，并标明“未确认”。
- 视频只通过 OpenCV 读取，不重新编码、不写临时文件。
- 判定规则、数据库迁移和视频搜索由 `desktop/tests/test_v13_video.py` 测试。

**已知限制：**
- **视频分析只能在桌面应用中进行，而且需要 GPU 环境。** 视频阶段由桌面应用的分析进程执行，该进程固定在 GPU 上运行 OCR；`index_cli.py analyze` 只分析照片。
- 视频识别规则来自开发阶段的可行性验证。验证所用的真实视频和证据数据不在本仓库中，对应的复现测试在没有这些数据时会自动跳过，所以**本仓库中没有用真实视频验证过的准确率数据**。
- 每个视频只抽取 5 到 7 帧，与视频长短无关。如果同一视频中有多个号码各得至少 2 票，结果为 AMBIGUOUS；票数不够时为 UNINDEXED。这两种情况都不会作为确认匹配。

## 系统要求

- **Windows 64 位**，以及 **Python 3.11**（64 位）。哈希锁定的依赖文件和 `.bat` 脚本都只针对这个平台。Linux 和 macOS **未测试**（最小测试环境见 [TESTING.md](TESTING.md)）。
- **CPU 与 GPU：**
  - 命令行可以在 **CPU** 上完成照片分析：`--ocr-device cpu` 是默认值。
  - **桌面应用的分析（照片和视频）需要 NVIDIA GPU 环境**，检测不到时“开始”按钮不可用。
  - 只搜索已有索引则不需要 GPU。
  - 人体检测（YOLO）默认在 CPU 上运行；GPU 环境只用于 OCR。
- AI 相关软件包下载量约 1–2 GB，另外还需下载模型文件（见 [MODEL_SETUP.md](MODEL_SETUP.md)）。

已测试的配置：Windows 10（x64）、Python 3.11.9、NVIDIA RTX 3060（CUDA 13.0、cuDNN 9.13）。这是测试过的配置，不是最低要求。

## 安装与启动

以下步骤摘自 [INSTALLATION.md](INSTALLATION.md)。所有命令在“命令提示符”和 PowerShell 中都能运行，不需要激活虚拟环境。

> 2026-10-17 的验证只用核心锁定文件和桌面锁定文件重新安装过环境（见下文“测试”）。完整引擎环境的安装（`setup_windows.bat` 不带参数）、模型下载，以及用命令行完整分析照片的流程，这次都**没有重新运行**。

### 1. 安装 Python 3.11
从 <https://www.python.org/downloads/windows/> 安装 64 位 Python 3.11，并勾选 **py launcher**。检查：
```
py -3.11 --version
```

### 2. 引擎环境（`.venv`，CPU）
在仓库文件夹中运行：
```
setup_windows.bat
```
脚本会：
1. 用 Python 3.11 创建 `.venv`；
2. 按 `requirements-lock-win-py311.txt` 安装依赖（`pip install --require-hashes --no-deps`，每个文件都校验 SHA-256）；
3. 运行 `pip check` 和 `check_environment.py`。

如果只需要运行单元测试和集成测试（Pillow、PyYAML、pytest，不含 AI）：
```
setup_windows.bat core
```

### 3. 下载模型
```
.\.venv\Scripts\python.exe check_environment.py --models
```
- 首次运行会下载 YOLO26n 和 PaddleOCR 模型，并在一张生成的测试图上各运行一次。这只检查模型能否加载和运行，**不衡量准确率或速度**。
- 模型保存在 `models\`（PaddleOCR 模型在 `models\paddlex\`），不在仓库中。已测试版本的文件大小和 SHA-256 见 [MODEL_SETUP.md](MODEL_SETUP.md)。
- 如果 PaddleOCR 下载失败，可以先设置 `PADDLE_PDX_MODEL_SOURCE=bos`（PowerShell：`$env:PADDLE_PDX_MODEL_SOURCE = "bos"`），再重新运行。

### 4. 用命令行分析照片（CPU）
```
.\.venv\Scripts\python.exe index_cli.py analyze "D:\RacePhotos\MyRace2026" --event "My Race 2026"
```
索引保存在 `data\runner_photo_finder.db`。其他命令见 `python index_cli.py --help`。

### 5. 桌面应用（可选，需单独的环境）
桌面界面使用单独的环境，目前没有脚本，需要手动创建：
```
py -3.11 -m venv .venv-desktop
.venv-desktop\Scripts\python.exe -m pip install --require-hashes --no-deps -r requirements-desktop-lock-win-py311.txt
.venv-desktop\Scripts\pythonw.exe desktop\RunnerFinder.pyw
```
- 分析任务在单独的“工作进程”中运行，按以下顺序选择 Python：
  1. 环境变量 `RUNNERFINDER_WORKER_PYTHON`；
  2. 仓库内的 `.venv-gpu\Scripts\python.exe`；
  3. 桌面环境自身的 Python。它不含 AI 组件，无法进行分析。
- 这个桌面环境的安装步骤**尚未在全新克隆上验证过**。

### 6. GPU 环境（可选）
GPU OCR 需要 `paddlepaddle-gpu` 3.3.1 以及 NVIDIA CUDA 13 / cuDNN 9 的 wheel 包。**本仓库目前无法复现 GPU 环境的安装**：`requirements-gpu-win-py311.txt` 只记录了测试环境的版本，不是锁定文件，也没有记录 `paddlepaddle-gpu` 的软件包源。环境准备好以后，可以用下面的命令检查：
```
<gpu python> index_cli.py gpu-check
```
批处理文件 `analyze_gpu.bat`、`ocr_gpu.bat` 使用环境变量 `RPF_GPU_PYTHON`，未设置时使用 `.venv-gpu\Scripts\python.exe`。

## 测试

详见 [TESTING.md](TESTING.md)。

```
run_tests.bat
run_tests.bat -m unit
run_tests.bat -m integration
```

桌面测试（使用桌面环境）：
```
.venv-desktop\Scripts\python.exe -m unittest desktop.tests.test_skeleton desktop.tests.test_v12_search desktop.tests.test_v13_gui -v
```

**最近一次验证**：2026-10-17，提交 `9cc298d`，Windows 10、Python 3.11.9，使用由锁定文件新装的环境。

| 测试 | 结果 |
|---|---|
| 单元测试 + 集成测试 | 369 通过，1 跳过（符号链接需要 Windows 开发者模式） |
| 桌面独立测试（3 个模块） | 23 通过 |
| `desktop.tests.test_v13_video` | 24 通过，2 跳过（需要私有数据） |
| 其他桌面模块（无私有数据、无 GPU 环境时） | 27 通过，43 跳过 |
| `tests/e2e`（真实 AI 组件） | **未运行**：环境中没有 AI 组件和模型 |
| GPU、私有数据和真实照片测试 | **未运行** |
| Linux / macOS | **未测试** |

**测试的局限：**
- 这些测试验证代码是否按设计工作，**不衡量真实比赛照片上的号码识别准确率**。
- 部分桌面测试依赖不在仓库中的私有数据（参考照片集、已有索引等），通过环境变量提供（见 `desktop/tests/private_data.py`），数据缺失时会自动跳过。
- 其中一些测试的断言针对特定数据集（照片数量、耗时、GPU 型号），换用其他数据时可能失败。

## 隐私与网络

- 照片和视频不会离开你的电脑。
- 只有安装软件包和首次下载模型时需要联网。
- `models\yolo26n.pt` 存在后，Ultralytics 以离线模式运行（`YOLO_OFFLINE=1`，在 `config.py` 中设置），其使用统计（`sync`）也会被关闭（`detection/person_detector.py`）。有一个 e2e 测试会在检测期间阻断所有网络连接，只要尝试联网就会失败。
- Streamlit 界面只监听 `localhost`，并关闭了使用统计（`.streamlit/config.toml`）。

## 许可证与第三方组件

Copyright (C) 2026 the RunnerFinder authors.

RunnerFinder 是自由软件：你可以依据自由软件基金会发布的 GNU Affero 通用公共许可证（AGPL）第 3 版，或（由你选择）任何更新的版本，重新分发和/或修改它。本程序**不提供任何担保**。许可证全文（英文）见 [LICENSE](LICENSE)，以英文原文为准。

- 本仓库**只包含 RunnerFinder 自己的源代码**，不包含第三方源代码、二进制文件或模型权重。依赖通过 requirement 文件从 PyPI 安装，模型在首次使用时下载。各组件遵循各自的许可证，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。主要组件：Ultralytics 与 YOLO26n 权重（AGPL-3.0）、PaddleOCR / PaddlePaddle 与 PP-OCRv5 模型（Apache-2.0）、PySide6（LGPL-3.0 或 GPL，或 Qt 商业许可）、OpenCV（Apache-2.0，wheel 中内含按 LGPL-2.1 授权的 FFmpeg）。
- 可选的 GPU 环境使用 NVIDIA CUDA/cuDNN 库，它们按 NVIDIA 的专有许可分发，也不在本仓库中。
- **关于 Windows 安装包：** 本仓库不提供安装包。打包了上述组件的安装包还有额外义务（许可证文本、源代码提供、NVIDIA 再分发条款等），这些义务仍在审查中，**现有的 Windows 安装包尚未获准公开再分发**。

## 已知限制

- **GPU 环境无法从本仓库复现**：桌面应用的分析（包括全部视频分析）因此只能在已有 GPU 环境的电脑上使用。没有 GPU 环境时，请用命令行在 CPU 上分析照片。
- **导出功能尚未实现**：`export/exporter.py` 只生成目标文件夹结构，`export_photos()` 会抛出 `NotImplementedError`。Streamlit 界面中的 Export 页面也只显示示例路径。
- **识别准确率未在本仓库中衡量**：准确率评估脚本 `evaluation/evaluate.py` 尚未实现（目前只输出一条提示），仓库也不附带标注数据或评估结果。
- **只在 Windows 上测试过**，见上文“系统要求”。

## 参与贡献与问题反馈

- **报告问题**：请在 GitHub Issues 中提交：<https://github.com/FakeChef/RunnerFinder/issues>。请写明 Windows 和 Python 版本、使用的是命令行还是桌面应用、CPU 还是 GPU，并附上完整的错误信息（`check_environment.py` 的输出通常很有帮助）。
- **请不要在 Issue 中上传真实的比赛照片、包含个人信息的截图，或你的索引数据库**（`data\` 目录）。
- **提交代码前**，请先运行单元测试和集成测试（`run_tests.bat -m unit`、`run_tests.bat -m integration`），并保持每个文件原有的换行符（`.bat` 文件使用 CRLF）。
- 本项目以 AGPL-3.0-or-later 授权；你提交的修改也将按同一许可证发布。
- 代码结构、配置项、环境变量和依赖文件的说明见 [DEVELOPMENT.md](DEVELOPMENT.md)（英文）。
