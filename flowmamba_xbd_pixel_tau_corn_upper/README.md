# VMamba xBD Pixel Tau CORN Upper

这是一个独立的新工程，用于 xBD 的像素级建筑损伤分类 upper-bound 测试。

当前项目的唯一正式主干是：

- `VMamba-S` 预训练骨干
- 像素级 `localization` head
- 像素级 `CORN` ordinal damage head
- conservative safe tau
- polygon pooling instance auxiliary supervision

这不是普通 softmax 分割项目，也不是旧的 oracle-instance upper-bound classification 项目。

## 项目定位

- 主监督来自 `post_disaster_target.png`
- `post_disaster.json` / WKT polygon 只用于训练时的实例一致性辅助监督
- 评测主线保留 xBD 风格指标：
  - `F1_loc`
  - `F1_subcls`
  - `F1_bda`
  - `F1_oa`

## Backbone 策略

- 正式实验只支持 `backend=vmamba`
- `backend=fallback` 仅用于本地 smoke test / 调试
- 不再支持 `FlowMamba` 作为本项目的正式 backbone 选项
- 不再支持 `auto` 或 “FlowMamba / VMamba 二选一” 语义
- 当配置要求 `VMamba` 时，若 VMamba 不可用，会明确报错，不会静默 fallback

说明：

- 仓库目录名里仍保留 `flowmamba`，这是历史命名遗留
- 若你本机使用 `conda run -n flowmamba`，这里只是环境名遗留，不表示本项目仍把 FlowMamba 当正式后端

## 当前配置约定

核心模型配置在 `configs/model_vmamba_tau_corn.yaml`：

- `model.backbone_name: vmamba_small`
- `model.backend: vmamba`
- `model.use_fallback_backbone: false`
- `model.fail_if_vmamba_unavailable: true`
- `model.pretrained_path: /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/pretrained/vssm_small_0229_ckpt_epoch_222.pth`

当前配置分工：

- `configs/exp_step1.yaml`
  - 最小正式 VMamba 闭环
- `configs/exp_step2_vmamba.yaml`
  - step2 VMamba 实验
- `configs/exp_fulltrain_vmamba.yaml`
  - fulltrain VMamba 实验
- `configs/exp_step1_fallback_smoke.yaml`
  - 显式 fallback 调试配置，只用于 smoke test

## 运行身份与 checkpoint

训练、验证、smoke test 启动时都会明确记录：

- 请求后端 `requested_backend`
- 实际后端 `resolved_backend`
- `fallback_used`
- `pretrained_path`
- `pretrained_loaded`
- `resolved_config_path`
- 当前 resolved config

checkpoint metadata 会保存：

- `backbone_name`
- `backend_name`
- `fallback_used`
- `pretrained_path`
- `pretrained_loaded`
- `resolved_config_path`

如果当前配置要求 `VMamba`，但 checkpoint 实际来自 `fallback` run，`validate` / `resume` 会直接报不兼容错误。

## 数据说明

xBD 当前使用方式：

- `post_disaster_target.png`
  - `0 = background`
  - `1 = no-damage`
  - `2 = minor-damage`
  - `3 = major-damage`
  - `4 = destroyed`
- `loc_target = post_target > 0`
- `damage_rank_target`
  - building pixels 上把 `{1,2,3,4}` 映射到 `{0,1,2,3}`
  - 背景填 `ignore_index`
- `post_disaster.json`
  - 解析 polygon / subtype
  - 只用于 polygon pooling instance auxiliary supervision
  - 不替代像素级 PNG 主监督

## 运行命令

准备 list：

```bash
python /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/tools/prepare_xbd_lists.py \
  --config /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/configs/exp_step1.yaml
```

正式 VMamba step1 训练：

```bash
CONDA_ENV=flowmamba bash /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/run/train.sh \
  /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/configs/exp_step1.yaml
```

正式 VMamba 验证：

```bash
CONDA_ENV=flowmamba bash /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/run/val.sh \
  /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/outputs/exp_step1_vmamba/checkpoints/latest.pth \
  /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/configs/exp_step1.yaml
```

fallback smoke test：

```bash
bash /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/run/smoke_test.sh \
  /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/configs/exp_step1_fallback_smoke.yaml
```

更正式的 VMamba 实验：

```bash
CONDA_ENV=flowmamba bash /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/run/train.sh \
  /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/configs/exp_step2_vmamba.yaml
```

```bash
CONDA_ENV=flowmamba bash /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/run/train.sh \
  /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/configs/exp_fulltrain_vmamba.yaml
```

## 输出目录

典型输出：

```text
outputs/
├── exp_step1_vmamba/
├── exp_step1_fallback_smoke/
├── exp_step2_vmamba/
└── exp_fulltrain_vmamba/
```

每个实验目录下通常包含：

- `logs/`
- `checkpoints/`
- `validation/`
- `debug_vis/`（smoke test 或调试可视化）
