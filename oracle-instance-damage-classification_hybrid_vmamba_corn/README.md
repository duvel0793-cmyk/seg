# Oracle Instance Damage Classification with Hybrid Conv-VMamba CORN

This project is a standalone upper-bound testbed for xBD / xView2 oracle instance-level damage classification.

It keeps the same task semantics as the reference `oracle-instance-damage-classification_corn` project:

- one sample = one GT building instance
- post-disaster JSON provides the real building polygon and real damage subtype
- pre/post crops are cut from the original imagery using the GT instance crop box
- the oracle instance mask is provided as input
- the output is 4-class instance-level damage classification:
  - `no-damage`
  - `minor-damage`
  - `major-damage`
  - `destroyed`

This is not a pixel-level damage segmentation project, not a building segmentation project, and not a BDA dense prediction project.

## What Changed vs Reference

Reference project:

- shared ResNet18 siamese encoder
- classification-style trunk for instance damage classification

This project:

- keeps the same classification-style trunk
  - shared siamese encoder
  - multi-scale fusion
  - channel attention / gating
  - masked multi-scale pooling
  - final instance-level classifier
- replaces the encoder with a shallow-conv + deep-VMamba hybrid encoder
- keeps the main training target focused on:
  - `model_type: oracle_mcd_corn`
  - `loss_mode: corn_adaptive_tau_safe`
- does not switch to a BDA-style temporal trunk
- does not add ChangeMamba-style dual-temporal dense interaction paths

## Hybrid Encoder

`HybridConvVMambaEncoder` uses:

- `stem`: shallow convolution, 4-channel input support
- `stage1`: convolutional residual blocks, output `c2`
- `stage2`: convolutional residual blocks, output `c3`
- `stage3`: VMamba-style blocks, output `c4`
- `stage4`: VMamba-style blocks, output `c5`

Default feature channels:

- `c2=96`
- `c3=192`
- `c4=384`
- `c5=768`

Design motivation:

- shallow convolution handles local texture, edges, and mask-aware boundary cues
- deep VMamba blocks provide stronger high-level semantics and longer-range spatial mixing
- the task semantics remain pure oracle instance upper-bound classification for apples-to-apples comparison

## VMamba Weights

The project searches under `/home/lky/data` for candidate VMamba / VSSM checkpoints and copies the selected weight into the project.

Actual selected source:

- `/home/lky/data/vssm_small_0229_ckpt_epoch_222.pth`

Project-local copy:

- `/home/lky/code/oracle-instance-damage-classification_hybrid_vmamba_corn/checkpoints/vmamba_pretrained.pth`

Default config path in the project:

- `model.pretrained: true`
- `model.vmamba_pretrained_weight_path: /home/lky/code/oracle-instance-damage-classification_hybrid_vmamba_corn/checkpoints/vmamba_pretrained.pth`

Loader behavior:

- supports checkpoint payloads under `state_dict`, `model`, or direct parameter dict
- strips common prefixes such as `module.`, `backbone.`, `encoder.`
- uses `strict=False`
- prints loaded / missing / unexpected keys
- expands 3-channel input weights to 4 channels when needed
  - the 4th channel is initialized from the mean of the first 3 channels
- if the weight path is empty or missing, loading is skipped with a warning and training still runs with random initialization

## Configs

Main configs:

- `configs/default.yaml`
- `configs/vscode_run.yaml`
- `configs/hybrid_vmamba_corn_safe.yaml`

All three keep the same xBD data paths as the reference project:

- `/home/lky/data/xBD`
- `/home/lky/data/xBD/xBD_list/train_all.txt`
- `/home/lky/data/xBD/xBD_list/val_all.txt`

## Run

Train with default config:

```bash
python train.py
```

Train with the explicit main experiment config:

```bash
python train.py --config configs/hybrid_vmamba_corn_safe.yaml
```

Train with an explicit project-local pretrained weight:

```bash
python train.py --vmamba_pretrained_weight_path /home/lky/code/oracle-instance-damage-classification_hybrid_vmamba_corn/checkpoints/vmamba_pretrained.pth
```

Evaluate a checkpoint:

```bash
python evaluate.py --config configs/hybrid_vmamba_corn_safe.yaml --resume /path/to/checkpoint.pth
```

## Notes

- The original reference repo was copied first and was not modified in place.
- This repo remains a full standalone directory for direct training and evaluation.
- Evaluation stays instance-level and keeps the original comparison-oriented metric outputs, including macro F1, weighted F1, balanced accuracy, per-class metrics, QWK, EMD severity, adjacent/far error, tau stats, and ordinal error profile.
