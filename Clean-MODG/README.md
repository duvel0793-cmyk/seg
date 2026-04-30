# Clean-MODG

Clean-MODG 是一个从零搭建的 PyTorch 科研工程，用于 xBD 数据集上的实例级建筑损伤有序分类。项目主线强调干净、可复现、可消融、便于后续论文扩展。

当前默认主线为：

`CleanDualScaleDamageNet = tight/context dual-scale + shared ConvNeXtV2 + explicit pre/post change + concat MLP fusion + CORN ordinal head`

## 1. 项目简介

本项目不做像素级损伤分割，而是面向单栋建筑实例的损伤等级预测。输入是灾前图、灾后图、建筑 polygon 或 bbox，经过 tight crop 和 context crop 后，由双尺度共享 backbone 模型输出建筑损伤等级。

## 2. 任务定义

输入：

- `pre-disaster image`
- `post-disaster image`
- `building polygon / instance mask / bbox`
- 基于同一实例构建的 `tight crop` 与 `context crop`

输出：

- 单栋建筑损伤等级 `0/1/2/3`

## 3. 类别定义

固定类别顺序如下：

1. `0 = no_damage`
2. `1 = minor_damage`
3. `2 = major_damage`
4. `3 = destroyed`

训练、验证、测试全过程都使用这一个 class order：

`["no_damage", "minor_damage", "major_damage", "destroyed"]`

## 4. manifest CSV 格式

训练完全由 manifest CSV 驱动，dataset 内不硬编码数据路径。CSV 至少需要以下列：

| column | meaning |
|---|---|
| `pre_image` | 灾前图像路径 |
| `post_image` | 灾后图像路径 |
| `label` | 0/1/2/3 |
| `polygon` | JSON string，格式 `[[x1,y1], [x2,y2], ...]`，可为空 |
| `bbox` | JSON string，格式 `[x_min, y_min, x_max, y_max]`，polygon 为空时回退 |
| `building_id` | 建筑实例 id |
| `disaster_id` | 灾害 id |
| `split` | `train / val / test` |

仓库默认提供了一个空表头文件：

`data/manifest.csv`

## 4.1 项目内置主干权重

项目目录已经包含默认 backbone 的本地预训练权重：

- [weights/convnextv2_tiny_22k_224_ema.pt](/home/lky/code/Clean-MODG/weights/convnextv2_tiny_22k_224_ema.pt)

默认配置中的 `model.pretrained_path` 已指向该文件，因此主线训练会优先从本地加载，不依赖在线下载。

## 5. 如何生成 manifest

如果你的 xBD 目录接近官方结构：

```bash
python data/build_xbd_manifest.py \
  --xbd-root /home/lky/data/xBD \
  --output /home/lky/code/Clean-MODG/data/manifest.csv
```

脚本会扫描类似结构：

```text
xBD/
  train/
    images/
    labels/
  test/
    images/
    labels/
```

说明：

- 当前脚本优先解析 `*_post_disaster.json` 中的 building polygon 和 damage subtype。
- 不同 xBD 变体的 JSON 结构可能略有不同。
- 如果你的版本与脚本不完全匹配，请直接手工生成 manifest，只要列名和格式满足本 README 即可。
- 如果没有现成 `val` split，建议你在生成后的 CSV 中手工拆分一部分 `train` 为 `val`。

## 6. 如何检查数据

检查 manifest：

```bash
python tools/check_manifest.py --manifest data/manifest.csv
```

分析分布：

```bash
python tools/analyze_dataset.py --manifest data/manifest.csv
```

检查一个 batch 并导出 tight/context 可视化：

```bash
python tools/check_batch.py --config configs/clean_dual_scale.yaml
```

## 7. 如何训练 clean model

```bash
python train.py --config configs/clean_dual_scale.yaml
```

或者：

```bash
bash scripts/train_clean.sh
```

## 8. 如何训练 hierarchical model

```bash
python train.py --config configs/hier_dual_scale.yaml
```

或者：

```bash
bash scripts/train_hier.sh
```

`HierDualScaleDamageNet` 默认开启 binary auxiliary head，并使用：

`loss = corn_loss + lambda_binary * binary_ce_loss`

## 9. 如何跑 ablation

顺序执行一组消融：

```bash
python run_ablation.py
```

只打印命令：

```bash
python run_ablation.py --print-only
```

当前提供的配置包括：

- `configs/ablate_tight_only.yaml`
- `configs/ablate_context_only.yaml`
- `configs/ablate_local_window_attention.yaml`
- `configs/ablate_neighborhood.yaml`
- `configs/ablate_ce_loss.yaml`
- `configs/ablate_focal_loss.yaml`
- `configs/ablate_mask_pooling.yaml`
- `configs/debug_overfit.yaml`

注意：

- `ablate_neighborhood.yaml` 会明确报错，因为主线工程没有默认实现 neighborhood branch。这是有意设计，避免 silent fail。

## 10. 如何测试

验证：

```bash
python validate.py \
  --config configs/clean_dual_scale.yaml \
  --checkpoint outputs/clean_dual_scale/checkpoints/best_macro_f1.pth
```

测试并导出 `predictions.csv`：

```bash
python test.py \
  --config configs/clean_dual_scale.yaml \
  --checkpoint outputs/clean_dual_scale/checkpoints/best_macro_f1.pth
```

单样本调试：

```bash
python infer_single.py \
  --config configs/clean_dual_scale.yaml \
  --checkpoint outputs/clean_dual_scale/checkpoints/best_macro_f1.pth \
  --pre-image /path/to/pre.png \
  --post-image /path/to/post.png \
  --bbox "[100,120,180,220]"
```

## 11. 如何可视化错误案例

随机可视化 crop：

```bash
python tools/visualize_crops.py \
  --manifest data/manifest.csv \
  --output-dir outputs/visualize_crops
```

可视化典型错误样本：

```bash
python tools/visualize_errors.py \
  --predictions outputs/clean_dual_scale/predictions/test_predictions.csv \
  --manifest data/manifest.csv
```

从 predictions 导出混淆矩阵：

```bash
python tools/export_confusion_matrix.py \
  --predictions outputs/clean_dual_scale/predictions/test_predictions.csv \
  --output outputs/clean_dual_scale/confusion/test_confusion.png
```

## 12. 当前默认主模型结构

默认模型 `CleanDualScaleDamageNet` 的 forward 主线非常简洁：

1. shared backbone 编码 `pre_tight / post_tight / pre_context / post_context`
2. tight 分支构建 explicit change feature
3. context 分支构建 explicit change feature
4. tight/context 分别池化为实例向量
5. `concat + MLP` 进行双尺度融合
6. 主 head 使用 `CORN ordinal regression`
7. 可选 binary auxiliary head 只在 hierarchical 版本启用

tight crop 主要关注建筑主体损伤细节，context crop 主要关注周边灾害环境语义。

## 13. 哪些模块默认关闭

默认不启用：

- `neighborhood`
- `alignment`
- `DamageAwareChangeBlock`
- `heatmap branch`
- `Adaptive Tau`
- `complex fusion`
- `LocalWindowAttention`

这些模块只允许通过 config 作为消融或后续扩展打开，不参与主线默认 forward。

## 14. 默认数据设计说明

- polygon 为空时自动回退到 bbox
- bbox 也为空时样本会被过滤或报清晰错误
- pre/post 使用完全一致的 crop 参数
- train 阶段采用同步增强
- val/test 阶段只做 resize + normalize
- 支持 `mask_mode = crop_only / mask_pooling / rgbm / none`

其中：

- `mask_pooling` 是默认主线
- `crop_only` 只用几何信息裁剪，不启用 mask-guided pooling
- `rgbm` 会把 mask 作为第 4 个通道，此时请把 `model.in_chans` 设为 `4`
- `none` 为 bbox-only/no-mask 风格实验接口

## 15. 训练输出

默认训练目录类似：

```text
outputs/clean_dual_scale/
  config.yaml
  log.txt
  metrics.csv
  checkpoints/
    best_macro_f1.pth
    best_bridge_score.pth
    last.pth
  confusion/
    epoch_xxx.png
  predictions/
    val_predictions.csv
    test_predictions.csv
```

训练日志会输出：

- train loss
- val loss
- accuracy
- macro F1
- per-class F1
- ordinal MAE
- severe error rate
- bridge score
- confusion matrix

## 16. Bridge Score 说明

`metrics/bridge_score.py` 当前实现的是 project-level placeholder。

它不是官方 Bridge Pixel Evaluation Score。

接口：

```python
compute_bridge_score(preds, targets, areas=None)
```

当前默认行为：

- 无 `areas` 时回退到 `macro_f1`
- 有 `areas` 时混合 `macro_f1` 与 area-weighted accuracy

正式论文、报告或对外结果汇报前，必须替换为官方 evaluator。

当前训练阶段建议优先关注以下指标：

- `macro F1`
- `per-class F1`
- `ordinal MAE`
- `severe error rate`

如果你有官方 Bridge Pixel Evaluation Score 评测脚本，请直接替换这个函数实现即可，其他训练/验证入口不需要改。

## 17. 后续扩展方向

- `mask perturbation`
- `bbox-only`
- `SAM / automatic instance generator`
- `BRIGHT multimodal extension`

## 18. 重要工程约束总结

- 主线默认模型保持干净，不堆复杂模块
- 所有关键行为都由 YAML config 控制
- 不依赖 wandb
- 单卡 GPU 可运行
- 支持 AMP mixed precision
- backbone 通过 `timm.create_model` 构建
- 若 `convnextv2_tiny` 不可用，会自动 fallback 到 `convnext_tiny` 并给出 warning
- 默认优先读取项目内 `weights/` 下的本地 ConvNeXtV2 权重
