# Oracle Instance Damage Classification with VMamba BDA CORN

This project is an independent oracle-instance upper-bound damage classification repo for xBD.

It keeps the original instance-level task semantics:

- one sample = one GT building instance
- inputs = `pre crop`, `post crop`, `oracle instance mask`
- label source = post-disaster json `properties.subtype`
- supervision = instance-level damage label only
- evaluation = instance-level macro F1 / weighted F1 / balanced accuracy / QWK / EMD severity / ordinal diagnostics

It does **not** switch the task to pixel-level BDA segmentation:

- not whole-image input
- not `target.png` supervision
- not damage segmentation map prediction
- not ChangeMamba’s original pixel-level BDA head

## What Changed vs the Reference Project

The reference project used a more classification-style oracle trunk.

This project changes the representation path to a BDA-style dual-temporal setup:

- shared dual-branch VMamba encoder
- four-scale features `c2/c3/c4/c5`
- per-scale temporal interaction with pre/post, absolute difference, and sum features
- lightweight temporal gates and change-aware fusion
- mask-aligned multi-scale pooling to produce an instance feature vector
- final instance-level standard or CORN head

Supervision is still strictly instance-level.

## Relation to ChangeMamba

This repo borrows the **dual-temporal BDA modeling paradigm** only:

- shared encoder for pre/post
- multi-scale temporal interaction
- BDA-style temporal trunk

It does **not** reproduce ChangeMamba’s pixel-level supervision task.
The final output here is an instance-level ordinal classifier, not a segmentation map.

## Model Overview

Input tensors:

- `pre_crop`: `[B, 3, H, W]`
- `post_crop`: `[B, 3, H, W]`
- `instance_mask`: `[B, 1, H, W]`

The model constructs:

- `pre_input = cat([pre_crop, instance_mask], dim=1)` -> 4 channels
- `post_input = cat([post_crop, instance_mask], dim=1)` -> 4 channels

Then it applies:

1. shared VMamba encoder on pre/post
2. per-scale BDA temporal fusion blocks
3. masked multi-scale pooling aligned with the oracle instance mask
4. instance-level head with full tau-aware CORN support

Main model types:

- `oracle_bda_vmamba_standard`
- `oracle_bda_vmamba_corn`

Default recommendation:

- `model.model_type: oracle_bda_vmamba_corn`

## VMamba Backbone and Pretrained Weights

The repo provides a local hierarchical VMamba-style backbone in [models/vmamba_basic.py](/home/lky/code/oracle-instance-damage-classification_vmamba_bda_corn/models/vmamba_basic.py).

Supported variants:

- `vmamba_tiny`
- `vmamba_small`

Weight loading is controlled by:

- `model.pretrained`
- `model.vmamba_pretrained_weight_path`

Example:

```yaml
model:
  backbone: vmamba_small
  pretrained: true
  vmamba_pretrained_weight_path: /path/to/your_vmamba_weight.pth
```

If the checkpoint patch embedding expects 3-channel input but this project uses 4-channel input, the loader expands the first layer automatically:

- channels 1-3 keep the original pretrained weights
- channel 4 is initialized with the mean of the first 3 channels

An empty path is handled safely and skipped.

## Configs

Main config files:

- `configs/default.yaml`
- `configs/vscode_run.yaml`
- `configs/corn_safe_full_recipe_vmamba_bda.yaml`

Recommended full recipe:

- `configs/corn_safe_full_recipe_vmamba_bda.yaml`

## Run

Train with default config:

```bash
python train.py
```

Train with the full VMamba+BDA+tau+CORN recipe:

```bash
python train.py --config configs/corn_safe_full_recipe_vmamba_bda.yaml
```

Train with another VMamba backbone and explicit pretrained weights:

```bash
python train.py --backbone vmamba_small --vmamba_pretrained_weight_path /path/to/your_vmamba_weight.pth
```

Evaluate a checkpoint:

```bash
python evaluate.py --config configs/corn_safe_full_recipe_vmamba_bda.yaml --resume /path/to/checkpoint.pth
```

## Notes

- The dataset, loss, metrics, and stage2 logic are maintained locally in this repo.
- There are no cross-project imports to the reference repo.
- The project keeps the full tau + CORN safe recipe, including dual-view, contrastive, soft ordinal regularization, and stage2 head rebalance.
