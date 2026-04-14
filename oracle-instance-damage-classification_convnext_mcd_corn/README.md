# oracle-instance-damage-classification_convnext_mcd_corn

本项目基于 `oracle-instance-damage-classification_corn` 中较稳的 `oracle_mcd + corn_adaptive_tau_safe` 主线思路重写，只保留 xBD 真实 json oracle 实例级损伤分类所需代码。

主要变化是把旧项目的 `ResNet18` 主干替换为 `ConvNeXt-Tiny`，同时保留旧主线里真正有价值的结构约束：

- shared siamese pre/post encoder
- pre/post 多尺度融合
- mask-aware multi-scale pooling
- 4 类辅助 CE 分支
- CORN ordinal head
- bounded safe adaptive tau

这里没有改成简单的 RGB-only ConvNeXt。旧项目稳定性的一个关键来自 oracle 实例 mask 的持续参与，所以当前实现默认采用：

- `3` 通道官方 ConvNeXt stem
- 每个 stage 后做 `mask gating`
- 多尺度 `masked avg + masked max pooling`

这样既保留 mask-aware MCD 主线，又不破坏官方 `ConvNeXt-Tiny` 预训练 stem 权重加载。另提供可选 `use_4ch_stem=true` 模式。

默认 loss 只保留主线：

- auxiliary CE
- CORN loss
- safe adaptive tau regularization

没有携带旧项目里无关的历史实验分支、CDA/UCL 变体、ResNet encoder、像素级分割逻辑。

## 目录

项目结构与实现模块如下：

- `datasets/`: xBD json polygon 解析、crop、mask、paired transform
- `models/`: ConvNeXt backbone、mask gating encoder、fusion、pooling、heads
- `losses/`: CORN 与 adaptive tau safe
- `metrics/`: 分类指标、CORN threshold decode、threshold calibration
- `engine/`: trainer / evaluator
- `scripts/`: dataset 检查、smoke test、与旧项目对比

## 训练

```bash
python train.py --config configs/default.yaml
```

训练过程会打印：

- dataset class counts
- sampler 权重
- ConvNeXt 预训练加载报告
- loss components
- train / val raw 和 calibrated macro-F1
- per-class F1
- best thresholds

输出默认保存到：

```text
outputs/default/
```

其中包括：

- `latest.pth`
- `best.pth`
- `metrics_history.json`
- `best_metrics.json`

## 评估

```bash
python evaluate.py --config configs/default.yaml --checkpoint outputs/default/best.pth
```

会输出：

- raw metrics
- calibrated metrics
- confusion matrix
- per-class precision / recall / F1

## 推理

```bash
python infer.py --config configs/default.yaml --checkpoint outputs/default/best.pth --split val --index 0
```

## 自检

```bash
python scripts/inspect_dataset.py --config configs/default.yaml
python scripts/smoke_test.py --config configs/default.yaml
```

如果本地没有预训练权重，项目会尝试自动下载：

```text
https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
```

下载失败不会崩溃，会回退到随机初始化并给出 warning。

## 当前范围

当前项目只保留主线实现，不包含旧项目无关实验分支。

