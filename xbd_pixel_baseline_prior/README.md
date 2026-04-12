# xBD Pixel Baseline Prior

轻量级 xBD pixel-level building damage assessment baseline，目标是作为后续所有像素级 damage 实验的统一起点：

- 对齐像素级评测口径
- 训练成本显著低于重型 VMamba / FlowMamba 路线
- 结构清晰，方便后续插入 SAM3 prior、CORN、轻量 tau、Tiny Mamba backbone、compact alignment module
- 第一版只保留最必要组件：ResNet18 backbone、轻量 FPN decoder、damage/localization 双头、可选 prior

## Project Layout

```text
xbd_pixel_baseline_prior/
  README.md
  requirements.txt
  train.py
  evaluate.py
  infer.py
  configs/
  src/
```

## Local Data Layout

本项目已经按本机真实目录适配过，不假设根目录下存在统一的 `targets/`。

当前本地数据结构：

```text
/home/lky/data/xBD/
  train/
    images/
    targets/
  test/
    images/
    targets/
  xBD_list/
    train_all.txt
    val_all.txt
```

实际样本映射规则：

- `sample_id -> {image_dir}/{sample_id}_pre_disaster.png`
- `sample_id -> {image_dir}/{sample_id}_post_disaster.png`
- `sample_id -> {split_target_dir}/{sample_id}_post_disaster_target.png`

当前本机上已验证：

- `train_all.txt -> /home/lky/data/xBD/train/images + /home/lky/data/xBD/train/targets`
- `val_all.txt -> /home/lky/data/xBD/test/images + /home/lky/data/xBD/test/targets`

说明：

- `configs/base.yaml` 里 `data.target_dir` 设为 `/home/lky/data/xBD`
- 代码会自动回退查找 `train/targets` 或 `test/targets`
- 启动训练/评估前会打印数据自检信息

## Damage Labels

- `0`: background
- `1`: no-damage
- `2`: minor-damage
- `3`: major-damage
- `4`: destroyed

## Prior Mask Layout

`data.prior_dir` 默认是 `null`，不会写死。

推荐的 prior 目录结构：

```text
/path/to/prior_dir/
  hurricane-florence_00000004.png
  guatemala-volcano_00000003.png
  ...
```

推荐命名规则是 `{sample_id}.png`，每张 prior 都是与样本一一对应的二值建筑掩码 png。代码也兼容少量常见别名，例如：

- `{sample_id}_post_disaster.png`
- `{sample_id}_post_disaster_mask.png`
- `{sample_id}_building_mask.png`

如果 `prior_mode` 不是 `none` 且找不到 prior 文件，会直接报错，不会静默跳过。

## Install

```bash
pip install -r requirements.txt
```

## Configs

### `configs/baseline_no_prior.yaml`

- `prior_mode: none`
- 输入为 `pre RGB + post RGB = 6` 通道
- 不读取 prior

### `configs/baseline_with_prior.yaml`

- `prior_mode: input_channel`
- 输入为 `pre RGB + post RGB + prior = 7` 通道
- 使用前先把 `data.prior_dir` 改成你的 prior 掩码目录

## Train

无 prior baseline：

```bash
python train.py --config configs/baseline_no_prior.yaml
```

带 prior 输入通道 baseline：

```bash
python train.py --config configs/baseline_with_prior.yaml
```

训练时会保存：

- `latest.pth`
- `best_macro_f1.pth`
- `metrics.json`
- `config_backup.yaml`
- `run.log`

## Evaluate

验证集评估：

```bash
python evaluate.py \
  --config configs/baseline_no_prior.yaml \
  --checkpoint /home/lky/code/xbd_pixel_baseline_prior/outputs/baseline_no_prior/best_macro_f1.pth
```

如果你后续有单独 test list / image dir，也可以显式指定：

```bash
python evaluate.py \
  --config configs/baseline_no_prior.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --list-path /path/to/test_list.txt \
  --image-dir /path/to/test/images \
  --target-dir /home/lky/data/xBD
```

可选保存预测图：

```bash
python evaluate.py \
  --config configs/baseline_no_prior.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --save-pred-masks \
  --save-overlay
```

## Infer

单样本推理，`--input` 可以传一张 xBD 风格的 `*_post_disaster.png` 或 `*_pre_disaster.png`，脚本会自动找到配对图：

```bash
python infer.py \
  --config configs/baseline_no_prior.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --input /home/lky/data/xBD/test/images/hurricane-florence_00000004_post_disaster.png \
  --output-dir /home/lky/code/xbd_pixel_baseline_prior/outputs/infer_no_prior
```

目录推理：

```bash
python infer.py \
  --config configs/baseline_no_prior.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --input /home/lky/data/xBD/test/images \
  --output-dir /home/lky/code/xbd_pixel_baseline_prior/outputs/infer_dir
```

带 prior 输入通道的目录推理：

```bash
python infer.py \
  --config configs/baseline_with_prior.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --input /home/lky/data/xBD/test/images \
  --output-dir /home/lky/code/xbd_pixel_baseline_prior/outputs/infer_with_prior \
  --prior-dir /path/to/prior_dir
```

推理会保存：

- `pred_masks/*.png`
- `overlays/*.png`

同时打印：

- 输入尺寸
- 输出尺寸
- 各类别像素统计

## Metrics

训练和评估会汇报：

- overall pixel accuracy
- per-class precision / recall / F1 / IoU
- No Damage F1
- Minor Damage F1
- Major Damage F1
- Destroyed F1
- Damage Macro F1（只统计 1~4）
- Building localization F1（由 `pred > 0` 与 `gt > 0` 计算）

## Model Summary

- Backbone: `torchvision resnet18`
- Input:
  - `none`: 6 channels
  - `input_channel`: 7 channels
  - `loss_weight`: 6 channels
- Decoder: custom lightweight FPN-style decoder
- Heads:
  - damage head: `[B, 5, H, W]`
  - localization aux head: `[B, 1, H, W]`

损失：

- damage loss: class-weighted focal cross entropy
- loc loss: BCEWithLogits + soft dice
- total loss: `damage_loss + lambda_loc * loc_loss`

## Extension Points

后续可以在这个 baseline 上继续扩展：

- CORN
- 轻量 tau
- Tiny Mamba backbone
- compact alignment module

## Notes

- `baseline_with_prior.yaml` 在 `data.prior_dir` 还是 `null` 时不能直接训练
- 如果你当前还没准备 prior，先跑 `configs/baseline_no_prior.yaml`
- 不修改任何上游 SAM3 代码，prior 只通过独立目录接入
