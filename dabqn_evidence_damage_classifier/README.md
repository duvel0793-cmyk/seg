# DABQN-lite for xBD

`dabqn_evidence_damage_classifier` 是在原 `evidence_hierarchical_damage_classifier` 基础上新建的独立项目，目标是把原先的 `oracle GT polygon -> instance damage classification` 上界设定，推进到 `predicted footprint -> instance damage classification` 的更真实设定。

## 与原 baseline 的关系

原 baseline:

- GT polygon / oracle mask
- tight / context / neighborhood 三尺度裁剪
- ConvNeXtV2 + scale fusion
- CORN / hierarchical ordinal damage classification

新项目:

- pre / post full image
- shared ConvNeXtV2 + FPN + query decoder
- 自动预测 building queries, boxes, masks
- predicted query 进入三尺度 damage branch
- CORN / hierarchical ordinal damage classification

这不是替代原 baseline，而是把 oracle footprint upper-bound 扩展到 predicted-footprint realistic setting。

## 当前实现

- Shared `ConvNeXtV2-Tiny/Small` backbone
- `FPN + pixel decoder`
- `learnable building queries`
- `Hungarian matching`
- `mask / bbox / objectness` localization losses
- `three-scale damage branch`
  - `mask_pool`: 默认，直接在 FPN feature 上做 tight/context/neighborhood mask-guided pooling
  - `crop_3scale`: 备选，在 shared feature space 上做 box-guided tight/context/neighborhood crop
- `CORN / hierarchical / CE` 三种 damage head
- `stage1 localization / stage2 damage / stage3 joint` 三阶段训练
- localization, matched-only damage, end-to-end damage, pixel bridge 四类评估

## 目录

- `configs/`: 训练与验证配置
- `datasets/`: xBD query dataset, polygon rasterize, transforms, collate
- `models/`: backbone, FPN, query decoder, matcher, damage branch, DABQN-lite
- `losses/`: mask, box, ordinal, total losses
- `engine/`: trainer 与 evaluator
- `metrics/`: localization, instance damage, pixel bridge metrics
- `utils/`: checkpoint, logger, seed, misc, EMA, scheduler
- `scripts/`: 三阶段训练与验证脚本

## 数据路径

默认配置使用:

- `data.root: /home/lky/data/xBD`
- `train_list: /home/lky/data/xBD/xBD_list/train_all.txt`
- `val_list: /home/lky/data/xBD/xBD_list/val_all.txt`
- `test_list: /home/lky/data/xBD/xBD_list/test_all.txt`

如果你的 xBD 路径不同，请优先修改 `configs/*.yaml`。

## 三阶段训练

Stage 1, 只训练 building localization:

```bash
cd /home/lky/code/dabqn_evidence_damage_classifier
python3 train_dabqn.py --config configs/stage1_localization.yaml --stage localization
```

Stage 2, 冻结大部分 localizer，训练 damage head:

```bash
cd /home/lky/code/dabqn_evidence_damage_classifier
python3 train_dabqn.py --config configs/stage2_damage.yaml --stage damage
```

Stage 3, joint finetuning:

```bash
cd /home/lky/code/dabqn_evidence_damage_classifier
python3 train_dabqn.py --config configs/stage3_joint.yaml --stage joint
```

也可以直接运行:

```bash
bash scripts/train_stage1_localization.sh
bash scripts/train_stage2_damage.sh
bash scripts/train_stage3_joint.sh
```

## 验证

```bash
cd /home/lky/code/dabqn_evidence_damage_classifier
python3 validate_dabqn.py \
  --config configs/stage3_joint.yaml \
  --checkpoint outputs/stage3_joint/checkpoints/best_bridge_score.pth \
  --split val
```

或:

```bash
bash scripts/validate_dabqn.sh
```

验证会输出:

- localization precision / recall / F1
- matched-only damage accuracy / macro F1 / per-class F1
- end-to-end instance damage macro F1 / weighted F1
- pixel bridge xView2-style overall score

## 迁移说明

直接复用了原项目中较稳定的工程模块和思路:

- `ConvNeXtV2` 权重加载逻辑
- `EMA`
- `WarmupCosineScheduler`
- `checkpoint` 原子保存
- xBD polygon 解析与 rasterize 核心逻辑
- bridge pixel evaluation 指标定义
- CORN / hierarchical ordinal 分类头思路

但数据输入与模型主线已经切换成 query-based instance localization。

## 最小 smoke test

安装依赖后，先做语法与最小前向检查:

```bash
cd /home/lky/code/dabqn_evidence_damage_classifier
python3 -m compileall .
python3 - <<'PY'
import torch
from utils.misc import load_yaml
from models import build_dabqn_model
cfg = load_yaml('configs/stage1_localization.yaml')
model = build_dabqn_model(cfg)
batch = {
    'pre_image': torch.randn(1, 3, cfg['dataset']['image_size'], cfg['dataset']['image_size']),
    'post_image': torch.randn(1, 3, cfg['dataset']['image_size'], cfg['dataset']['image_size']),
}
out = model.forward_localization(batch)
print(out['pred_logits'].shape, out['pred_boxes'].shape, out['pred_masks'].shape)
PY
```

再做 dataset smoke test:

```bash
cd /home/lky/code/dabqn_evidence_damage_classifier
python3 - <<'PY'
from utils.misc import load_yaml
from datasets import XBDQueryDataset
cfg = load_yaml('configs/stage1_localization.yaml')
ds = XBDQueryDataset(config=cfg, split='val', is_train=False)
item = ds[0]
print(item['pre_image'].shape, item['post_image'].shape, item['target']['masks'].shape, item['target']['boxes'].shape)
PY
```

## 如果后续加入 direct polygon decoder

建议优先修改这些文件:

- `models/query/building_query_decoder.py`
- `models/query/matcher.py`
- `models/dabqn_lite.py`
- `losses/dabqn_losses.py`
- `metrics/localization_metrics.py`
- `metrics/pixel_bridge_metrics.py`
- `engine/evaluator.py`

因为 polygon decoder 会同时改变 query 输出形式、matching cost、localization metric 和 rasterization 入口。
