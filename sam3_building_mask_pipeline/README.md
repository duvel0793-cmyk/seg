# sam_compare

基于 xBD `train` split 的 supervised SAM3 建筑二值分割对比实验项目。项目读取 xBD 的灾前图像 `*_pre_disaster.*` 和对应的灾前建筑 GT mask `*_pre_disaster_target.png`，对接本地 `sam3` 上游代码做训练/微调，并在 `test` split 上导出预测 mask 与基础像素级指标。

## 项目特点

- 独立于其他历史工程，不依赖 `building_segmentation` 或 `sam3_building_prior`
- 优先复用本地 `sam3` 的真实可 import 接口
- 不修改上游 `sam3` 源码，所有耦合集中在 `sam_compare/model_adapter.py`
- 默认基于目录扫描构建数据集，兼容 xBD 中 `0/1` 或 `0/255` mask
- 输出 `pred_masks/*.png`、`metrics.json`、`per_image_metrics.csv`

## 目录结构

```text
sam_compare/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── train.yaml
│   └── eval.yaml
├── outputs/
│   └── .gitkeep
├── scripts/
│   ├── train_xbd_sam3.py
│   ├── eval_xbd_sam3.py
│   └── run_train_eval.py
├── sam3/
└── sam_compare/
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── export.py
    ├── infer.py
    ├── metrics.py
    ├── model_adapter.py
    ├── paths.py
    ├── postprocess.py
    ├── trainer.py
    └── utils.py
```

## 环境

项目假设你已经有可用的 `sam3` 环境。当前机器上检测到一个现成环境名为 `sam3`，推荐先进入它：

```bash
conda activate sam3
```

然后在项目根目录运行脚本。项目不会联网下载权重，`sam3` 权重路径需要本地可访问。

## 路径配置

默认路径集中在：

- [configs/default.yaml](/home/lky/code/sam_compare/configs/default.yaml)
- [sam_compare/paths.py](/home/lky/code/sam_compare/sam_compare/paths.py)

其中默认值包括：

- xBD 根目录：`/home/lky/data/xBD`
- 上游 `sam3` 目录：当前项目下的 `sam3/`
- 输出根目录：当前项目下的 `outputs/`

`sam3` checkpoint 可以通过以下方式指定：

1. 命令行 `--checkpoint /path/to/sam3.pt`
2. 在 [configs/default.yaml](/home/lky/code/sam_compare/configs/default.yaml) 中填写 `paths.checkpoint`
3. 设置环境变量 `SAM3_CHECKPOINT`
4. 把 `sam3.pt` 放到项目根目录、`weight/` 或 `weights/` 目录，代码会自动搜索常见位置

## 数据约定

代码已经按 xBD 的真实目录结构适配：

- `train/images`, `train/labels`, `train/targets`
- `test/images`, `test/labels`, `test/targets`

默认行为：

- 只保留 `*_pre_disaster.*`
- mask 使用 `*_pre_disaster_target.png`
- `labels/*.json` 仅作为配对校验与元数据来源，不会误用 post-disaster damage label
- mask 二值化兼容 `0/1` 和 `0/255`
- 当前 adapter 按上游骨干的真实约束固定使用 `1008x1008` 输入尺寸

## 训练

```bash
python scripts/train_xbd_sam3.py \
  --checkpoint /path/to/sam3.pt \
  --output-dir outputs/xbd_sam3_train \
  --device cuda \
  --epochs 5 \
  --batch-size 1 \
  --lr 1e-4 \
  --num-workers 4
```

训练输出包括：

- `output_dir/checkpoints/`
- `output_dir/logs/train.log`
- `output_dir/train_config.json`
- `output_dir/training_summary.json`
- `output_dir/best_model.pth`
- `output_dir/last_model.pth`

说明：

- 这里的 `--checkpoint` 默认表示 SAM3 基础权重
- 训练时会从 `train` split 内部按固定随机种子切出可复现 validation 子集，默认 `val_ratio=0.1`、`split_seed=42`
- `best_model.pth` 改为基于 validation `IoU` 选择，`last_model.pth` 仍然保留最后一个 epoch
- 默认 loss 为 `weighted BCE + soft Tversky`，其中 BCE 会对前景和 GT 边界区域加权，并对极小前景样本做轻度 boost
- `resume` 用于恢复本项目训练中断点，而不是基础 `sam3.pt`

## 测试集推理与评估

```bash
python scripts/eval_xbd_sam3.py \
  --checkpoint outputs/xbd_sam3_train/best_model.pth \
  --output-dir outputs/xbd_sam3_eval \
  --device cuda \
  --batch-size 1 \
  --num-workers 4 \
  --save-pred-masks
```

评估输出包括：

- `output_dir/pred_masks/*.png`
- `output_dir/metrics.json`
- `output_dir/metrics_breakdown.json`
- `output_dir/per_image_metrics.csv`
- `output_dir/eval_config_used.json`
- `output_dir/postprocess_config_used.json`
- `output_dir/threshold_sweep.json`（如果启用 threshold sweep）
- `output_dir/tta_used.json`（如果启用 TTA）

默认评估仍然输出原图尺寸上的建筑二值 mask，但现在支持：

- 在 validation 子集上做轻量 threshold sweep
- horizontal flip / vertical flip / `rot90` TTA
- 轻量 small component removal / small hole filling 后处理

如果 `--checkpoint` 指向的是本项目训练产物 `best_model.pth`，评估会自动加载完整模型状态；如果它指向的是原始 `sam3.pt`，则只会加载骨干权重，分割头将是随机初始化状态，日志会明确警告该结果通常不具可比性。

## 一键串联训练 + 评估

```bash
python scripts/run_train_eval.py \
  --checkpoint /path/to/sam3.pt \
  --output-dir outputs/xbd_sam3_run \
  --device cuda \
  --epochs 5 \
  --batch-size 1 \
  --lr 1e-4 \
  --num-workers 4
```

一键脚本会：

1. 在 `train/` 子目录训练
2. 使用训练得到的 `best_model.pth` 在 `test/` 子目录评估
3. 生成 `summary.json`

## 真实依赖的上游 SAM3 接口

本项目不是伪造 API，当前实现真实依赖了上游的：

- `sam3.build_sam3_image_model`
- `SAM3VLBackbone.forward_image` 输出的 `backbone_fpn`

也就是说，这个项目当前走的是：

`SAM3 图像骨干特征 -> 轻量二值分割头 -> xBD building mask`

这是为了满足“单张 pre-disaster image 直接输出单通道 building mask”的监督训练目标，同时保持对上游源码零侵入。

另外，adapter 会在训练时把上游 `vitdet` 的 fused MLP 快路径回退到标准 PyTorch 前向，以兼容梯度计算；这一步是在 wrapper 中完成的，没有改动上游源码文件。

## 备注

- 当前 shell 默认 Python 没有自动进入带 `torch` 的环境，请先 `conda activate sam3`
- 如果你后续把 `sam3.pt` 放到新的本地路径，请同步更新命令行参数或默认配置
- 需要手动确认的本地路径主要是：`sam3.pt` 的真实位置，以及你希望使用的 GPU 设备
