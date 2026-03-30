# SAM3 Building Prior

`sam3_building_prior` 用于对 xBD 的 `pre-disaster` 图像生成建筑二值掩码，并整理成适合批量实验的三条主入口：

- `scripts/run_xbd_pre_mask.py`
  批量推理
- `scripts/evaluate_building_masks.py`
  批量评估
- `scripts/run_xbd_full_experiment.py`
  一键跑“批量推理 + 批量评估”

当前支持两条主实验路线：

- `text-only baseline`
- `baseline-guided refine`

其中 `refine` 依赖外部 coarse mask 目录，再基于 coarse 连通域生成 `box / point / box_point` prompts 做 SAM3 refinement。

## 目录结构

```text
sam3_building_prior/
  .vscode/
    launch.json
  checkpoints/
    sam3.pt
  sam3_building_prior/
    cli.py
    config.py
    dataset.py
    evaluation.py
    experiment.py
    io_utils.py
    postprocess.py
    prompting.py
    sam3_adapter.py
    types.py
    visualize.py
  scripts/
    run_xbd_pre_mask.py
    evaluate_building_masks.py
    run_xbd_full_experiment.py
  outputs/
  README.md
```

## 环境与默认路径

- 工作目录：`/home/lky/code/new/sam3_building_prior`
- 本地 `sam3` 仓库：`/home/lky/code/new/sam3`
- 默认 checkpoint：`./checkpoints/sam3.pt`
- 默认 xBD 根目录：`/home/lky/data/xBD`
- 推荐 Python：`/home/lky/anaconda3/envs/sam3/bin/python`

支持的 split：

- `train`
- `test`
- `hold`
- `tier3`

如果传：

```bash
--split train
```

输入图像会自动映射到：

```text
/home/lky/data/xBD/train/images
```

评估时默认 GT 会映射到：

```text
/home/lky/data/xBD/train/targets
```

如果同时传了 `--input-dir` 和 `--split`，以 `--input-dir` 为准。

## 输入约定

批量推理只会处理以下 `pre-disaster` 图像：

- `*_pre_disaster.png`
- `*_pre_disaster.jpg`
- `*_pre_disaster.jpeg`
- `*_pre_disaster.tif`
- `*_pre_disaster.tiff`

`--input-dir` 既可以是目录，也可以直接给单张 `pre-disaster` 图像路径。

`refine` 模式要求：

- 提供 `--coarse-mask-dir`
- coarse mask 文件名与输入图像同名，或至少同 stem 可匹配
- coarse mask 为单通道二值图

## 常用推理参数

`run_xbd_pre_mask.py` 主要参数：

- `--split`
- `--input-dir`
- `--output-dir`
- `--checkpoint`
- `--device`
- `--text-prompt`
- `--coarse-mask-dir`
- `--prompt-mode`
- `--prompt-min-area`
- `--min-area`
- `--limit`
- `--start-index`
- `--num-workers`
- `--overwrite`
- `--skip-existing`
- `--save-vis`
- `--save-json`
- `--save-prompts`
- `--log-file`

说明：

- `--prompt-mode text` 是 `text-only baseline`
- `--prompt-mode box / point / box_point` 是 `refine`
- `--num-workers` 仅用于扫描文件，不会开启多进程模型推理
- `--skip-existing` 会跳过已经产出完整目标文件的样本
- `--overwrite` 会先删掉该样本已有输出，再重新生成

## 推荐命令

### 1. Train 全量 text baseline

```bash
python scripts/run_xbd_pre_mask.py \
  --split train \
  --output-dir ./outputs/xbd_train_text \
  --checkpoint ./checkpoints/sam3.pt \
  --device cuda \
  --prompt-mode text \
  --text-prompt building \
  --skip-existing \
  --save-json \
  --save-vis \
  --save-prompts
```

### 2. Train 全量 refine

```bash
python scripts/run_xbd_pre_mask.py \
  --split train \
  --output-dir ./outputs/xbd_train_refine_box_point \
  --checkpoint ./checkpoints/sam3.pt \
  --device cuda \
  --coarse-mask-dir /path/to/coarse_masks \
  --prompt-mode box_point \
  --prompt-min-area 64 \
  --skip-existing \
  --save-json \
  --save-vis \
  --save-prompts
```

### 3. 批量评估

```bash
python scripts/evaluate_building_masks.py \
  --pred-dir ./outputs/xbd_train_refine_box_point/masks \
  --gt-dir /home/lky/data/xBD/train/targets \
  --summary-json ./outputs/xbd_train_refine_box_point/metrics_summary.json \
  --per-image-csv ./outputs/xbd_train_refine_box_point/per_image_metrics.csv
```

### 4. 一键全量实验

```bash
python scripts/run_xbd_full_experiment.py \
  --split train \
  --output-dir ./outputs/xbd_train_refine_box_point \
  --checkpoint ./checkpoints/sam3.pt \
  --device cuda \
  --coarse-mask-dir /path/to/coarse_masks \
  --prompt-mode box_point \
  --prompt-min-area 64 \
  --save-json \
  --save-vis \
  --save-prompts
```

### 5. 单图快速验证

```bash
python scripts/run_xbd_pre_mask.py \
  --input-dir /path/to/one_pre_disaster.png \
  --output-dir ./outputs/single_text \
  --checkpoint ./checkpoints/sam3.pt \
  --device cuda \
  --prompt-mode text \
  --text-prompt building \
  --save-json \
  --save-vis \
  --save-prompts
```

## 评估输出

`evaluate_building_masks.py` 支持：

- `--pred-dir`
- `--gt-dir`
- `--split`
- `--start-index`
- `--limit`
- `--per-image-csv`
- `--summary-json`
- `--verbose`

评估逻辑：

- 自动按同名文件匹配
- 如果同名不存在，会继续尝试 `*_pre_disaster_target.*`
- 只评估成功匹配到且尺寸一致的样本
- unmatched 样本会记到 `skipped_files`
- 读取失败或尺寸不一致的样本会记到 `failed_files`

汇总指标包括：

- `mean IoU`
- `mean F1`
- `mean Precision`
- `mean Recall`
- `matched images`
- `skipped images`
- `failed images`
- `evaluated_files`

## 输出目录

典型输出如下：

```text
outputs/
  xbd_train_text/
    masks/
    instances/
    visualizations/
    prompts/
    logs/
    summary.json
  xbd_train_refine_box_point/
    masks/
    instances/
    visualizations/
    prompts/
    logs/
    summary.json
    metrics_summary.json
    per_image_metrics.csv
    experiment_summary.json
```

各文件含义：

- `summary.json`
  推理阶段汇总，记录 success / failed / skipped
- `metrics_summary.json`
  评估阶段总体指标汇总
- `per_image_metrics.csv`
  每张图的 IoU / F1 / Precision / Recall
- `experiment_summary.json`
  一键实验的总汇总

## VSCode 一键运行

项目已提供：

```text
.vscode/launch.json
```

可直接从 VSCode 运行以下配置：

- `SAM3: Single Image Text`
- `SAM3: Train Text Baseline`
- `SAM3: Train Refine`
- `SAM3: Evaluate Train Refine`
- `SAM3: Full Experiment Refine`

使用前请确认 VSCode Python 解释器为：

```text
/home/lky/anaconda3/envs/sam3/bin/python
```

其中 `refine` 相关配置会在启动时提示输入 `coarse mask` 目录。

## 实验建议

- `train` 全量更适合工程验证、检查目录结构、观察可视化和排查失败样本
- 正式对比更建议放到 held-out / `test` split
- `refine` 的收益高度依赖 coarse mask 质量
