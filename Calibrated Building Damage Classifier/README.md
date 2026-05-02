# Calibrated Building Damage Classifier

Single-mainline instance-level building damage classification for xBD/xView2.

Each sample contains:

- `pre-disaster crop`
- `post-disaster crop`
- `oracle instance mask`
- one ordered label in `{no, minor, major, destroyed}`

## Mainline

Only one recommended model path is kept:

- shared `ConvNeXtV2 Tiny` backbone
- three-scale crops: `tight / context / neighborhood`
- lightweight `residual_gate` alignment
- lightweight `DamageAwareChangeBlock` with soft pseudo-change suppression
- single-path pre/post fusion
- per-scale local attention tokenization
- cross-scale attention over `tight / context / neighborhood`
- residual feature calibration before the `CORN` ordinal classifier
- `CORN` ordinal classifier with optional damage auxiliary / unchanged consistency / monotonic regularization

Removed from the runnable mainline:

- `diff_hotspot` local crop
- local fine-grained branch
- global + local fusion
- hotspot-driven local feature extraction
- old multi-branch ablation switches
- instance silhouette branch
- neighbor-instance token branch
- target-conditioned neighbor attention

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --config configs/default.yaml
```

Ablations:

```bash
python train.py --config configs/ablations/baseline.yaml
python train.py --config configs/ablations/full.yaml
```

## Evaluate

Instance-level evaluation:

```bash
python evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/default/checkpoints/final_best.pth \
  --mode instance
```

Export instance predictions:

```bash
python evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/default/checkpoints/final_best.pth \
  --mode export
```

Run pixel-level bridge evaluation:

```bash
python evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/default/checkpoints/final_best.pth \
  --mode bridge
```

## Layout

- `configs/default.yaml`: cleaned runtime config
- `datasets/xbd_oracle_instance_damage.py`: instance sample indexing and crop/mask generation
- `models/backbone.py`: shared ConvNeXtV2 backbone
- `models/alignment.py`: lightweight feature alignment
- `models/fusion.py`: aligned pre/post fusion
- `models/token_aggregator.py`: mask-aware evidence token aggregation
- `models/change_suppression.py`: lightweight damage-aware change modeling and pseudo-change suppression
- `models/classifier.py`: CORN ordinal head
- `models/model.py`: assembled single-path model
- `bridge/pixel_bridge.py`: instance-to-pixel projection and official-style scoring
- `bridge/evaluate_bridge.py`: bridge export orchestration
- `train.py`: training entrypoint
- `evaluate.py`: instance/export/bridge evaluation entrypoint
