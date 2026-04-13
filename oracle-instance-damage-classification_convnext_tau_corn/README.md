# Oracle Instance Damage Classification with ConvNeXt + Tau-CORN

本项目用于做 xBD 上的 `oracle / upper-bound` 实例级建筑损伤分类实验。

任务定义是：直接使用 xBD 原始 json 提供的真实建筑 polygon / subtype 标注作为实例定位来源，不做检测、不做分割预测。模型对每个真实建筑实例输入 `pre-disaster crop`、`post-disaster crop` 和 `oracle building mask`，输出 4 类有序损伤等级：

- `0 = no-damage`
- `1 = minor-damage`
- `2 = major-damage`
- `3 = destroyed`

## 项目目标

构建一个干净、稳定、可训练的 Python / PyTorch 项目，用于评估在 oracle 实例定位前提下的建筑损伤分类上界。

当前版本的核心设计：

- 共享权重双分支 `ConvNeXt`
- backbone 后使用 `oracle mask` 做 `masked global pooling`
- `pre/post/delta/abs(delta)` 特征拼接融合
- `learnable tau + CORN` 有序分类头

## 数据来源与假设

- 数据根目录固定为：`/home/lky/data/xBD`
- 当前实现优先适配本地实际结构：
  - `train/images`
  - `train/labels`
  - `test/images`
  - `test/labels`
  - `xBD_list/train_all.txt`
  - `xBD_list/val_all.txt`
- `train/val` 默认使用 `xBD_list/*.txt` 中的 tile id 构造 split
- `test` 默认扫描 `test/labels`
- parser 优先使用 `shapely.wkt`；若本机没有 `shapely`，则退化到仓库内置的轻量 WKT parser

## 目录结构

```text
oracle-instance-damage-classification_convnext_tau_corn/
├── README.md
├── train.py
├── evaluate.py
├── infer.py
├── configs/
├── models/
├── datasets/
├── losses/
├── engine/
├── metrics/
├── utils/
├── scripts/
└── outputs/
```

## 模型设计

### 1. Backbone

- 使用本仓库内实现的接近官方风格的 `ConvNeXt`
- 默认 `ConvNeXt-Tiny`
- 结构上支持切换到 `small / base`
- 不依赖 `timm`
- 不调用 `torchvision.models.convnext`

### 2. Oracle Masked Pooling

- `pre_image` 和 `post_image` 分别经过同一个共享权重 backbone
- `oracle building mask` 不进入 backbone stem
- 在 backbone 输出特征图上做 mask resize + masked average pooling

### 3. Fusion

- 使用：
  - `f_pre`
  - `f_post`
  - `f_post - f_pre`
  - `abs(f_post - f_pre)`
- 拼接后送入轻量 `MLP fusion head`

### 4. Learnable Tau + CORN

- 这是 4 类有序分类，不是普通 softmax
- 头部输出 `K-1 = 3` 个 ordinal logits
- 再通过正值、可学习、受约束的 `tau` 做稳定缩放
- 默认：
  - `tau_mode = per_threshold`
  - `final_logits = base_logits / tau`
- 训练损失：
  - `total_loss = corn_loss + tau_reg_weight * tau_center_reg + tau_diff_weight * tau_diff_reg`

## 使用环境

仓库默认运行在你现成的 `flowmamba` 虚拟环境中。

本仓库不负责环境配置，不创建 conda / pip 安装脚本。`requirements.txt` 只列出代码依赖范围，便于参考。

## 训练

```bash
cd /home/lky/code/oracle-instance-damage-classification_convnext_tau_corn
python train.py --config configs/default.yaml
```

## 评估

```bash
python evaluate.py --config configs/default.yaml --checkpoint /path/to/best.pth --split val
python evaluate.py --config configs/default.yaml --checkpoint /path/to/best.pth --split test
```

## 推理

```bash
python infer.py --config configs/default.yaml --checkpoint /path/to/best.pth --split val --num-batches 1
```

## Smoke Test

```bash
python scripts/smoke_test.py
```

该脚本会：

- 取一个 batch
- 前向运行主模型
- 打印关键张量 shape
- 打印 tau
- 打印 pred_labels

## Dataset Inspect

```bash
python scripts/inspect_dataset.py
python scripts/inspect_dataset.py --save-visuals
```

## 导出 split 统计

```bash
python scripts/export_split_stats.py
```

## 备注

- 本项目只做实例级建筑损伤分类
- 不做像素级 damage segmentation
- 不做光流
- 不做检测 / SAM / 多任务 / teacher-student
- backbone 保持纯 `ConvNeXt`

