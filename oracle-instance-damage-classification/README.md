# Oracle Instance Damage Classification

This project is a standalone PyTorch implementation for the xBD **GT-instance upper-bound damage classification** experiment.

It is intentionally **not**:

- a pixel-level damage segmentation project
- an instance generation / instanceization project
- a full FlowMamba reproduction
- a `target.png`-supervised damage map pipeline
- an MFA / optical-flow alignment system

Its question is narrower and cleaner:

> If the building instance is already known perfectly from the post-disaster GT polygon, how well can we classify its 4-way damage label?

That makes this repository an **oracle-instance upper-bound classifier**:

- each sample = one post-disaster GT polygon / one building instance
- input = `pre crop + post crop + oracle instance mask`
- output = one of:
  - `no-damage`
  - `minor-damage`
  - `major-damage`
  - `destroyed`

## Why This Stage Does Not Change The Backbone

The current research target is **ordinal supervision / boundary ambiguity modeling**, especially around `minor-damage` vs `major-damage`.

So this stage keeps the existing instance classifier backbone intact:

- paired crop construction stays the same
- shared ResNet18 Siamese encoder stays the same
- `BidirectionalFusionBlock` stays the same
- `ChannelAttentionGate` stays the same
- `MaskedMultiScalePooling` stays the same
- `oracle_mcd` stays the main trunk

The goal here is not to search for a stronger feature extractor. The goal is to test whether **ordinally-aware supervision and ambiguity-aware targets** can improve the oracle upper bound without redefining the task.

## Oracle Sample Definition

Each sample is **one post-disaster building polygon**.

- instance definition: the post-disaster json polygon
- label source: `properties.subtype`
- kept labels:
  - `no-damage`
  - `minor-damage`
  - `major-damage`
  - `destroyed`
- filtered label:
  - `un-classified`

For each polygon:

1. compute the tight bbox
2. expand with `context_ratio`
3. crop the same region from pre and post images
4. rasterize the post polygon as the oracle instance mask
5. predict one 4-class damage label for that instance

## Models

### `post_only`

- input: `post_rgb + instance_mask`
- role: appearance-only baseline

### `siamese_simple`

- input: `pre_rgb + post_rgb + instance_mask`
- role: simplest bi-temporal baseline

### `oracle_mcd`

Main classifier:

- shared ResNet18 Siamese encoder
- 4-channel branch input: `RGB + oracle mask`
- multi-scale `C2/C3/C4/C5`
- `BidirectionalFusionBlock` at every scale
- `ChannelAttentionGate`
- `MaskedMultiScalePooling`
- MLP classifier head

### `oracle_mcd_corn`

Strong ordinal baseline:

- reuses the same `oracle_mcd` encoder / fusion / pooling trunk
- only replaces the final classifier head with a rank-consistent CORN head
- does **not** change the backbone or the task

## Loss Modes

### `weighted_ce`

- class-weighted cross entropy baseline

### `fixed_cda`

- `L = L_ce + lambda_ord * L_fixed_cda`
- soft target distribution built from fixed ordinal distance by class index

### `learnable_cda`

- `L = L_ce + lambda_ord * L_learnable_cda + lambda_gap_reg * L_gap_reg`
- keeps class order fixed
- learns three adjacent severity gaps:
  - `gap_01`
  - `gap_12`
  - `gap_23`
- keeps a single global learnable `tau`

### `adaptive_ucl_cda`

Main upgrade in this revision:

- starts from the same learnable global severity axis as `learnable_cda`
- adds a lightweight sample-level `OrdinalAmbiguityHead`
- predicts one sample-specific `tau_i`
- builds sample-adaptive ordinal soft targets:
  - `q_ij = softmax(-|s[y_i] - s[j]| / tau_i)`
- adds:
  - unimodality regularization
  - concentration regularization
  - gap regularization

This keeps the classifier at instance level, but lets the supervision become wider on ambiguous samples and sharper on easier ones.

### `adaptive_ucl_cda_v3`

- keeps the same `oracle_mcd` trunk and ordinal severity axis
- `tau` now defaults to **bounded_sigmoid** parameterization:
  - `raw_tau = raw_tau_center + tau_logit_scale * tanh(raw_delta_tau)`
  - `tau = tau_min + (tau_max - tau_min) * sigmoid(raw_tau)`
- `raw_tau_center` is initialized from `tau_target` in logit-space, so the ambiguity head does not start from a hard-coded zero-logit prior
- residual `tau` is still available only for compatibility / ablation, but is no longer the recommended default
- the v3 objective now combines:
  - `tau_mean` regularization around `tau_target`
  - `tau_variance / std-floor` anti-collapse regularization
  - difficulty-guided `tau -> tau_ref` regression
  - direct `raw_tau -> raw_tau_ref` logit-space regression
  - batch-level `raw_tau_center` anchoring
  - soft `raw_tau` edge penalty to keep logits away from the bounded interval edges
  - optional pairwise `tau` rank regularization inside the batch

The raw-tau terms are there specifically to prevent the new collapse mode where `raw_tau` drifts to very large negative values, saturates the sigmoid input, and leaves only weak tau-space gradients near `tau_min`.

When checking whether adaptive `tau` is really working, prioritize these training logs / history fields:

- `tau_at_min_ratio` and `tau_at_max_ratio`
- `corr_tau_difficulty`
- `corr_raw_tau_difficulty`
- `raw_tau_edge_ratio`
- `tau_by_difficulty_bin`
- `raw_tau_minus_ref_stats`
- `tau_stats` and `raw_tau_stats`
- `tau_minus_tau_ref_stats`
- `tau_by_class`

### `corn`

Ordinal baseline for comparison:

- same oracle trunk as `oracle_mcd`
- final head predicts `K-1 = 3` threshold logits
- trained with standard **CORN** conditional ordinal regression
- exported together with threshold probabilities, 4-class probabilities, and severity expectation

This is the main hard baseline against the soft-label / CDA route.

### `corn_adaptive_tau_safe`

Conservative fusion mode:

- CORN remains the main classifier and still learns the primary ordinal boundaries from the standard conditional CORN loss
- the existing adaptive `tau` v3 branch is kept only as an auxiliary sample-wise ambiguity teacher in decoded 4-class probability space
- the auxiliary soft loss is delayed by default:
  - `corn_soft_start_epoch = 3`
- the adaptive soft target is detached by default before `loss_corn_soft`, so the soft supervision does not back-propagate into the ambiguity head
- training further constrains `loss_corn_soft` to update only the CORN head, reducing the risk of disturbing the shared trunk / CORN main boundary learning

When validating whether this conservative fusion is healthy, prioritize:

- `macro_f1`
- `weighted_f1`
- `far_error_rate`
- `per_class.minor-damage.f1`
- `per_class.major-damage.f1`
- `corr_tau_difficulty`
- `corn_soft_enabled`

## Evaluation Outputs

The evaluation pipeline keeps the previous classification metrics and adds ordinal-aware reporting:

- `macro_f1`
- `weighted_f1`
- `balanced_accuracy`
- per-class precision / recall / f1
- adjacency confusion
- severity error
- `quadratic_weighted_kappa`
- `emd_severity` (Wasserstein-1 / EMD on the ordered class distribution)
- `adjacent_error_rate`
- `far_error_rate`

Artifacts now include:

- `qwk.json`
- `emd.json`
- `ordinal_error_profile.json`
- `ordinal_error_profile.txt`
- `tau_stats.json`
- learned severity positions and reference soft target matrix
- hardest `minor-damage` / `major-damage` confusion visuals

## Config Priority And Direct Run

No-argument direct run is supported.

The default config path resolution is:

1. `configs/vscode_run.yaml`
2. `configs/default.yaml`

So if `configs/vscode_run.yaml` exists, bare `python train.py` and `python evaluate.py` will read that file first.

## Installation

```bash
pip install -r requirements.txt
```

## Main Experiments

All five requested experiment families are directly runnable.

### A. `oracle_mcd + weighted_ce`

```bash
python train.py --model_type oracle_mcd --loss_mode weighted_ce
python evaluate.py --model_type oracle_mcd --loss_mode weighted_ce
```

### B. `oracle_mcd + fixed_cda`

```bash
python train.py --model_type oracle_mcd --loss_mode fixed_cda
python evaluate.py --model_type oracle_mcd --loss_mode fixed_cda
```

### C. `oracle_mcd + learnable_cda`

```bash
python train.py --model_type oracle_mcd --loss_mode learnable_cda
python evaluate.py --model_type oracle_mcd --loss_mode learnable_cda
```

### D. `oracle_mcd + adaptive_ucl_cda`

```bash
python train.py --model_type oracle_mcd --loss_mode adaptive_ucl_cda
python evaluate.py --model_type oracle_mcd --loss_mode adaptive_ucl_cda
```

### E. `oracle_mcd_corn`

```bash
python train.py --model_type oracle_mcd_corn
python evaluate.py --model_type oracle_mcd_corn
```

`oracle_mcd_corn` automatically harmonizes to `head_type=corn` and `loss_mode=corn`, so you do not need to pass the loss manually unless you want to be explicit.

### F. `oracle_mcd_corn + corn_adaptive_tau_safe`

```bash
python train.py \
  --config configs/vscode_run.yaml \
  --loss_mode corn_adaptive_tau_safe \
  --lambda_corn_soft 0.03 \
  --corn_soft_start_epoch 3 \
  --early_stop_patience 6 \
  --save_topk 3
```

```bash
python train.py \
  --config configs/vscode_run.yaml \
  --loss_mode corn_adaptive_tau_safe \
  --lambda_corn_soft 0.05 \
  --corn_soft_start_epoch 3 \
  --early_stop_patience 6 \
  --save_topk 3
```

## Useful Overrides

```bash
python train.py --batch_size 16 --epochs 30 --image_size 256 --loss_mode adaptive_ucl_cda
python train.py --model_type oracle_mcd --loss_mode adaptive_ucl_cda --lambda_uni 0.10 --lambda_conc 0.05
python train.py --config configs/vscode_run.yaml --loss_mode corn_adaptive_tau_safe --lambda_corn_soft 0.03 --corn_soft_start_epoch 3 --early_stop_patience 6 --save_topk 3
python evaluate.py --checkpoint ./outputs/direct_run/oracle_mcd/adaptive_ucl_cda/checkpoints/best_macro_f1.pth
python evaluate.py --checkpoint ./outputs/direct_run/oracle_mcd_corn/corn/checkpoints/best_macro_f1.pth
python evaluate.py --checkpoint ./outputs/vscode_run/oracle_mcd_corn/corn_adaptive_tau_safe/checkpoints/best_macro_f1.pth
```

## Default Training Setup

- image size: `224`
- batch size: `32`
- epochs: `50`
- optimizer: `AdamW`
- lr: `3e-4`
- weight decay: `1e-4`
- warmup: `3` epochs

For `adaptive_ucl_cda`, the optimizer uses parameter groups:

- main model params: `lr = base_lr`, standard weight decay
- learnable ordinal gaps: `lr = 0.5 * base_lr`, `weight_decay = 0`
- ambiguity head params: `lr = 0.5 * base_lr`, `weight_decay = 0`

## Project Structure

```text
project_root/
├── configs/
│   ├── default.yaml
│   └── vscode_run.yaml
├── datasets/
│   ├── xbd_oracle_instance_damage.py
│   └── transforms.py
├── models/
│   ├── encoder.py
│   ├── fusion.py
│   ├── pooling.py
│   ├── heads.py
│   ├── baselines.py
│   └── oracle_mc_damage_model.py
├── utils/
│   ├── io.py
│   ├── geometry.py
│   ├── metrics.py
│   ├── losses.py
│   ├── visualize.py
│   ├── seed.py
│   └── cache.py
├── train.py
├── evaluate.py
└── README.md
```
