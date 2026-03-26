# Building Segmentation

这个目录把原项目里“建筑分割”相关的部分单独拆出来，目标是只做 `pre_disaster image -> building mask`。

## 目录结构

- `train.py`: 训练建筑分割模型
- `evaluate.py`: 评估 checkpoint，输出 `mIoU / IoU / F1 / Pixel Acc`
- `visualize_dataset.py`: 直接把 xBD 的建筑真值 mask 可视化出来
- `datasets/`: xBD 建筑分割数据集与增强
- `models/`: 建筑分割 backbone、decoder、network
- `utils/`: loss、metrics、评估和可视化工具
- `pretrained_weight/`: 放置 encoder 预训练权重

## 默认数据组织

默认读取 xBD 的以下结构：

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

其中：

- 建筑输入图像：`images/<name>_pre_disaster.png`
- 建筑真值 mask：`targets/<name>_pre_disaster_target.png`

## 训练

```bash
conda run -n flowmamba python building_segmentation/train.py \
  --train_dataset_path /home/lky/data/xBD/train \
  --train_data_list_path /home/lky/data/xBD/xBD_list/train_all.txt \
  --val_dataset_path /home/lky/data/xBD/test \
  --val_data_list_path /home/lky/data/xBD/xBD_list/val_all.txt \
  --save_dir building_segmentation/outputs/train_run
```

训练输出：

- `best_model.pth`
- `last_model.pth`
- `best_metrics.json`
- `metrics_epoch_xxx.json`
- `visualizations/epoch_xxx/`

## 评估

```bash
conda run -n flowmamba python building_segmentation/evaluate.py \
  --resume building_segmentation/outputs/train_run/best_model.pth \
  --dataset_path /home/lky/data/xBD/test \
  --data_list_path /home/lky/data/xBD/xBD_list/val_all.txt \
  --output_dir building_segmentation/outputs/eval_run
```

评估会输出：

- `metrics.json`
- `visualizations/*.png`

指标包括：

- `miou`
- `iou_background`
- `iou_building`
- `f1`
- `precision`
- `recall`
- `pixel_accuracy`

## 真值可视化

```bash
conda run -n flowmamba python building_segmentation/visualize_dataset.py \
  --dataset_path /home/lky/data/xBD/train \
  --data_list_path /home/lky/data/xBD/xBD_list/train_all.txt \
  --output_dir building_segmentation/outputs/gt_vis \
  --num_samples 4
```

## 说明

- 这个子项目只保留建筑分割，不再包含灾损分类分支。
- `models/vmamba.py` 和 `models/csm_triton.py` 已经内置在子项目里，不再依赖仓库里的 `classification/` 或 `changedetection/` 目录。
- 默认会从 `building_segmentation/pretrained_weight/vssm_small_0229_ckpt_epoch_222.pth` 读取 encoder 预训练权重；如果你放在别处，就显式传 `--pretrained_weight_path`。
- 训练和评估都要求 CUDA；当前这套 `vmamba` selective-scan kernel 不支持 CPU 前向。
- 如果当前 shell 环境没有安装 `torch`，训练和评估前还需要先切到你平时用的 Conda/PyTorch 环境。
