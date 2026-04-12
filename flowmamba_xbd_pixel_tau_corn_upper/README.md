# FlowMamba xBD Pixel Tau CORN Upper

## 项目目标

这是一个全新的、独立于旧工程的 xBD 像素级损伤分类上限测试项目。

主线设计：

- `VMamba / FlowMamba` 风格骨干
- 像素级 `localization` 分支
- 像素级 `CORN` 损伤排序主头
- 保守型可学习 `safe tau`
- 基于 `polygon pooling` 的实例一致性辅助监督

任务定位很明确：

- 主监督来自 `post_disaster_target.png`
- polygon / JSON 只用于训练时的实例一致性辅助监督
- 当前阶段目标是先把工程主干、接口、数据流、模型流、损失流、评测流和最小可运行闭环搭起来

## 本地检查结果

本地已检查到以下真实情况：

- 新项目目录：`/home/lky/code/flowmamba_xbd_pixel_tau_corn_upper`
- 本地参考仓库存在：
  - `/home/lky/code/FlowMamba/FlowMamba`
  - `/home/lky/code/VMamba`
- 当前 Python 环境里已安装：`torch / torchvision / PyYAML / Pillow / numpy / shapely`
- 当前 Python 环境里缺少若干上游依赖：`timm / mmengine / einops`
- 本机已存在可用 conda 环境：`flowmamba`、`sam3`

因此本项目的骨干接入策略是：

- 优先尝试接入本地真实 `FlowMamba / VMamba`
- 若环境依赖不足，则自动退回到项目内置的轻量 fallback backbone
- 接口保持与 FlowMamba 风格一致，后续补齐依赖后可继续替换为真实上游后端

当前已经确认：

- `conda run -n flowmamba` 可导入 `timm / einops / mmengine`
- `conda run -n flowmamba` 可导入本地 `FlowMamba` 与 `VMamba`
- VMamba 权重默认放在 `pretrained/vssm_small_0229_ckpt_epoch_222.pth`

## 当前第一步已完成内容

- 工程主干与目录结构
- YAML 配置系统
- 基于本地真实 xBD 结构的数据读取与 manifest 构建
- `post target -> loc target / damage rank target` 生成逻辑
- xBD polygon JSON / WKT 解析与裁剪同步
- `FlowMamba / VMamba` 风格骨干 wrapper
- `localization head + pixel CORN head + safe tau`
- `polygon pooling` 实例级辅助监督骨架
- 基础训练、验证、smoke test 脚本
- checkpoint / logger / metrics / confusion matrix
- 一次最小 `forward + backward + validation` 闭环

## 当前第一步尚未完成内容

- 更强的数据增强
- 更复杂的 tau 设计
- `feature pooling` 版实例辅助监督
- 严格对齐官方全部评测细节
- DDP、多卡、多阶段训练
- 更高级优化器与 AMP 策略调优
- 第二步的系统性完善与消融

## 本地真实数据结构说明

当前机器上的 xBD 不是根目录直接放 `targets/labels`，而是 split 结构：

```text
/home/lky/data/xBD/
├── train/
│   ├── images/
│   ├── targets/
│   └── labels/
├── test/
│   ├── images/
│   ├── targets/
│   └── labels/
├── hold/
├── tier3/
└── xBD_list/
    ├── train_all.txt
    └── val_all.txt
```

当前项目已经按这个真实结构实现：

- `train_all.txt` 默认映射到 `train/*`
- `val_all.txt` 默认映射到 `test/*`
- 若 list 缺失，可 fallback 扫描 split 目录
- 若未来出现根级 `images/targets/labels`，manifest 解析也预留了兼容逻辑

## 数据标签说明

`post_disaster_target.png` 的像素取值定义：

- `0 = background`
- `1 = no-damage`
- `2 = minor-damage`
- `3 = major-damage`
- `4 = destroyed`

训练时使用方式：

- `loc_target = post_target > 0`
- `damage_rank_target` 只在 building pixels 上有效
- `damage_rank_target` 把 `{1,2,3,4}` 映射到 `{0,1,2,3}`
- 背景位置填 `ignore_index`

polygon JSON 的用途：

- 来自 `post_disaster.json`
- 从 WKT 解析建筑 polygon 与 subtype
- 训练时在 polygon 内对像素级 ordinal logits 做 masked mean pooling
- 形成 instance-level CORN auxiliary loss
- 不替代像素级 target PNG 主监督

## 运行方式

建议先做数据检查：

```bash
python /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/tools/prepare_xbd_lists.py \
  --config /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/configs/exp_step1.yaml
```

smoke test：

```bash
bash /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/run/smoke_test.sh
```

若希望优先走真实 FlowMamba / VMamba 后端，建议：

```bash
CONDA_ENV=flowmamba bash /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/run/smoke_test.sh
```

训练：

```bash
bash /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/run/train.sh
```

或：

```bash
CONDA_ENV=flowmamba bash /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/run/train.sh
```

`configs/exp_step1.yaml` 当前默认是最小闭环配置：

- `1 epoch`
- `max_train_batches = 2`
- `max_val_batches = 1`

这样可以先稳定验证工程主干、损失与评测是否跑通，再继续往第二步扩展。

验证：

```bash
bash /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/run/val.sh \
  /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/outputs/checkpoints/latest.pth
```

或：

```bash
CONDA_ENV=flowmamba bash /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/run/val.sh \
  /home/lky/code/flowmamba_xbd_pixel_tau_corn_upper/outputs/checkpoints/latest.pth
```

## 结果与输出目录

```text
outputs/
├── logs/
├── checkpoints/
└── debug_vis/
```

其中：

- `logs/` 保存训练与验证日志、manifest 检查结果
- `checkpoints/` 保存 `latest` 与 `best`
- `debug_vis/` 保存 smoke test 可视化结果

## 备注

当前默认配置会优先尝试本地上游骨干，但在依赖不足时自动回落到内置 fallback backbone，以保证第一步最小闭环先能跑通。
