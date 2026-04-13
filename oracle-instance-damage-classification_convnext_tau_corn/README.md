# Oracle Instance Damage Classification with ConvNeXt + Tau-CORN

本项目用于做 xBD 原始 json 上的 `oracle / upper-bound` 实例级建筑损伤分类。

核心主线保持不变：

- shared-weight dual-branch `ConvNeXt` backbone
- `pre/post` 图像分别经过同一个 backbone
- `oracle mask` 只在 backbone feature map 后做 masked global pooling
- 融合特征固定为 `concat(f_pre, f_post, f_post - f_pre, abs(f_post - f_pre))`
- 最终头部为 `learnable tau + CORN`

当前版本额外支持：

- 自动联网下载 ConvNeXt 官方 Tiny 预训练权重
- 本地加载 backbone checkpoint
- train split 使用 weighted sampler 处理类别不平衡
- `per-threshold tau + CORN`

## Model Design

- Backbone: 仓库内官方风格 `ConvNeXt` 实现，不依赖 `timm`，不使用 `torchvision.models.convnext`
- Pooling: backbone 输出特征图后做 oracle masked global pooling
- Fusion: `f_pre / f_post / delta / abs(delta)` 拼接后进入 MLP fusion head
- Head: `K-1=3` 个 ordinal logits，随后由 learnable tau 做稳定缩放，`final_logits = base_logits / tau`

## ConvNeXt pretrained weights

默认支持自动下载 ConvNeXt Tiny 官方权重。

- 默认 URL: `https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth`
- 默认保存路径: `/home/lky/code/oracle-instance-damage-classification_convnext_tau_corn/pretrained/convnext/convnext_tiny_1k_224_ema.pth`
- 若本地已存在且文件大小大于 0，则不会重复下载
- 若下载失败，会打印 warning，并自动回退到随机初始化，不会让 `smoke_test` / 训练入口直接崩溃
- 加载时会自动跳过官方分类头 `head.*`

## Class imbalance handling

train split 可启用 `WeightedRandomSampler`。

- `use_weighted_sampler=true` 时，仅 train split 使用 sampler
- `sample_weight = 1 / freq[label] ** sampler_power`
- `sampler_power` 越大，长尾类别会被采样得更频繁
- val/test 不会强行启用 weighted sampler

## Tau mode

项目同时支持两种 tau 模式：

- `shared`: 所有 ordinal threshold 共用一个全局 tau
- `per_threshold`: 每个 ordinal threshold 拥有一个全局可学习 tau

当前默认是 `per_threshold`。对于 4 类 ordinal 任务，这意味着默认学习 3 个 tau，更贴近阈值建模。

## Config Defaults

默认模型配置位于 `configs/model/convnext_tiny_tau_corn.yaml`，其中关键项包括：

- `load_pretrained: true`
- `auto_download_pretrained: true`
- `tau_mode: per_threshold`
- `decode_mode: threshold_count`
- `freeze_backbone: false`

默认训练配置位于 `configs/runtime/train_runtime.yaml`，其中关键项包括：

- `batch_size: 8`
- `epochs: 30`
- `lr: 1e-4`
- `grad_clip: 5.0`
- `use_weighted_sampler: true`
- `sampler_power: 0.5`

## Train

```bash
cd /home/lky/code/oracle-instance-damage-classification_convnext_tau_corn
python train.py --config configs/default.yaml
```

训练启动时会打印：

- pretrained path / url
- 是否自动下载、是否加载、是否冻结 backbone
- tau mode
- weighted sampler 开关、类别计数、权重样例
- backbone pretrained report

## Evaluate

```bash
python evaluate.py --config configs/default.yaml --checkpoint /path/to/best.pth --split val
python evaluate.py --config configs/default.yaml --checkpoint /path/to/best.pth --split test
```

评估时会保持与训练相同的 `decode_mode`，并打印 backbone pretrained report、tau 统计、per-class F1 等信息。

## Smoke Test

```bash
python scripts/smoke_test.py --config configs/default.yaml
```

`smoke_test.py` 会打印：

- backbone variant
- pretrained path / url
- 是否尝试下载、是否下载成功
- `pretrained_loaded`
- loaded / skipped / missing / unexpected key 数量
- tau mode、tau raw、tau clamped
- logits shape、pred labels

若本地没有权重，会先尝试自动下载；若下载失败，则继续以随机初始化完成 smoke test。

## Notes

- 任务仍然是 oracle 实例级建筑损伤分类，不做检测或像素级 damage segmentation
- 不引入 VMamba / Mamba / SAM / Swin / ViT / teacher-student / sample-wise adaptive tau
- backbone 仍保持标准 3 通道输入，不改成 6 通道
