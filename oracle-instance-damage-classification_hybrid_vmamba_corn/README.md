# Oracle Instance Damage Classification Mainline

这个仓库现在只保留一条训练/评估主线，用于 xBD / xView2 的 oracle instance upper-bound damage classification。

任务定义保持不变：

- 一个 GT building instance = 一个样本
- 输入是 `pre crop + post crop + oracle instance mask`
- 输出是 4 类实例级 damage classification
- 仍然是 classification-style trunk，不是像素级训练，也不是 BDA-style trunk

## 唯一主线

- `backbone: hybrid_vmamba`
- `model_type: oracle_mcd_corn`
- `head_type: corn`
- `loss_mode: corn_adaptive_tau_safe`

模型主干保留：

- `HybridConvVMambaEncoder`
- pre/post shared siamese encoder
- classification-style multi-scale fusion
- `ChannelAttentionGate`
- `MaskedMultiScalePooling`
- CORN ordinal head
- safe adaptive tau ambiguity head

## 配置

仅保留两份主线配置：

- `configs/default.yaml`
- `configs/vscode_run.yaml`

两者都指向同一条主线。

## 训练

默认训练：

```bash
python train.py --config configs/default.yaml
```

常用覆盖项示例：

```bash
python train.py \
  --config configs/default.yaml \
  --batch_size 8 \
  --epochs 20 \
  --num_workers 4 \
  --vmamba_pretrained_weight_path /home/lky/code/oracle-instance-damage-classification_hybrid_vmamba_corn/checkpoints/vmamba_pretrained.pth
```

## 评估

```bash
python evaluate.py \
  --config configs/default.yaml \
  --resume /path/to/checkpoint.pth
```

## 评估输出

评估会保留实例级核心指标与导出文件，包括：

- `macro_f1`
- `weighted_f1`
- `balanced_accuracy`
- per-class metrics
- `quadratic_weighted_kappa`
- `emd_severity`
- `adjacent_error_rate`
- `far_error_rate`
- `tau_stats`
- `ordinal_error_profile`
