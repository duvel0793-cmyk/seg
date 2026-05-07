# Evidence Hierarchical Damage Classifier

该项目是一个全新的本地 xBD / xView2 风格实例级建筑损伤分类项目，目标是在保留旧项目稳定主线的前提下，引入两项新方法：`damage evidence localization` 与 `two-stage hierarchical ordinal classification`。

## 项目目标

输入为 `pre-disaster RGB`、`post-disaster RGB` 与目标建筑 polygon。输出为 4 类有序损伤标签：

- `0 no-damage`
- `1 minor-damage`
- `2 major-damage`
- `3 destroyed`

## 数据路径

默认数据根目录：

- `/home/lky/data/xBD`

默认列表文件：

- `train`: `/home/lky/data/xBD/xBD_list/train_all.txt`
- `val`: `/home/lky/data/xBD/xBD_list/val_all.txt`

## 模型结构

主线保留：

- 三尺度 crop：`tight / context / neighborhood`
- 每尺度输入：`pre RGB / post RGB / target mask`
- `ConvNeXtV2`
- `CrossScaleTargetConditionedAttention`
- `ResidualFeatureCalibration`
- `EMA`

新增方法：

- `damage evidence localization`
  每个尺度从 `fused feature map` 生成 `evidence_logits` 与 `severity_map`，并只在 target mask 内聚合 evidence statistics。
- `two-stage hierarchical ordinal classification`
  Stage A 判断 `damaged / no-damage`，Stage B 对 damaged 样本做 `minor / major / destroyed` 的 CORN ordinal 预测。

## 训练命令

```bash
python train.py --config configs/train_evidence_hier_corn.yaml
```

## Instance Eval

```bash
python evaluate.py \
  --config configs/train_evidence_hier_corn.yaml \
  --checkpoint outputs/evidence_hier_corn/checkpoints/best_ema_macro_f1.pth \
  --mode instance
```

## Bridge Eval

```bash
python evaluate.py \
  --config configs/train_evidence_hier_corn.yaml \
  --checkpoint outputs/evidence_hier_corn/checkpoints/best_ema_macro_f1.pth \
  --mode bridge
```

## Ablation

```bash
python train.py --config configs/train_ablation_no_evidence.yaml
python train.py --config configs/train_ablation_no_hierarchy.yaml
python train.py --config configs/train_fast_debug.yaml
```

## Oracle Mask Pixel Ordinal Line

这条实验线用于回答一个更聚焦的研究问题：

- 在真实建筑 `mask` 已知的条件下，把当前实例级有序分类思想迁移到 `mask` 内像素级 dense prediction，能达到什么上限。

它不是完整的端到端 building damage assessment：

- 不做 localization
- 不做自动实例分割
- 不替代现有 instance branch

监督方式是 weak pixel supervision：

- 建筑实例标签会被栅格化到该建筑 `mask` 内的所有有效像素
- `mask` 外像素全部忽略，不参与 pixel loss

训练命令：

```bash
python train.py --config configs/pixel_ordinal_oracle_line.yaml
```

像素线评估：

```bash
python tools/evaluate_pixel_ordinal_line.py \
  --config configs/pixel_ordinal_oracle_line.yaml \
  --checkpoint outputs/pixel_ordinal_oracle_line/checkpoints/best_ema_pixel_agg_macro_f1.pth \
  --split val \
  --use-ema
```

bridge 评估：

```bash
python evaluate.py \
  --config configs/pixel_ordinal_oracle_line.yaml \
  --checkpoint outputs/pixel_ordinal_oracle_line/checkpoints/best_ema_pixel_agg_macro_f1.pth \
  --mode bridge \
  --prediction-source pixel_agg
```

当前默认 bridge 使用 `pixel_agg`：

- 即先在 `mask` 内做 dense pixel prediction
- 再通过 mask-guided aggregation 得到实例级预测
- 最后把聚合后的实例标签回填整栋建筑 polygon

后续融合方向：

- `pixel_instance_probabilities` 可以和 `final_class_probabilities` 做概率融合
- `pixel` dense map 可以进一步作为 evidence map 或 consistency target

## Structural Fusion 建议实验顺序

```bash
python train.py --config configs/evidence_clean_v1_next_userideas.yaml

python tools/sweep_structural_fusion.py \
  --config configs/evidence_clean_v1_next_userideas.yaml \
  --checkpoint outputs/evidence_clean_v1_next_userideas/checkpoints/best_ema_macro_f1.pth

python train.py --config configs/evidence_clean_v1_userideas_struct_fuse01.yaml
python train.py --config configs/evidence_clean_v1_userideas_struct_fuse02.yaml
python train.py --config configs/evidence_clean_v1_userideas_struct_fuse03.yaml

python train.py --config configs/evidence_clean_v1_userideas_struct_only.yaml
python train.py --config configs/evidence_clean_v1_userideas_router_only.yaml
```

推荐顺序：

- 先继续训练 `evidence_clean_v1_next_userideas` 到 early stopping。
- 对 best checkpoint 跑 `tools/sweep_structural_fusion.py`，先做 inference-only structural fusion sweep。
- 如果 structural fusion 有收益，再正式训练 `struct_fuse01 / 02 / 03`。
- 最后跑 `struct_only` 和 `router_only`，判断 `Structural Two-Stage CORN` 与 `Severity-Aware Scale Router` 的独立贡献。

## 与旧项目区别

- 新项目与旧项目同级，代码全新整理，不污染旧项目。
- 保留旧项目稳定主线，但删除历史失败方向。
- 没有 `silhouette` 分支。
- 没有 `neighbor-aware token / graph consistency` 分支。
- `neighborhood` 仅作为第三个视觉尺度，不拆邻居 token。
