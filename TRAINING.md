# RecGen Training & Fine-tuning

This document covers training the two RecGen flow models from their TRELLIS
bases (reproducing the released checkpoints) and fine-tuning from the released
RecGen checkpoints on your own data.

The training code lives in the `recgen_training` package and reuses all model
architectures from `recgen_inference` — there is a single source of truth for
the networks, and any checkpoint written by training loads directly in the
inference pipeline.

**What is covered** — two flow models, each fine-tuned from the frozen
[TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large)
base (the TRELLIS VAEs/decoders stay frozen and are **not** trained):

- **Stage 1 — sparse structure**: `SparseStructurePoseFlowModel` +
  `MultiImageConditionedFlowMatchingCFGTrainer`
  → `stereo_denoiser_ema0.9999_step0055000.pt`
- **Stage 2 — structured latent (SLAT)**: `ElasticSLatCondFlowModel` +
  `MultiImageConditionedSparseFlowMatchingCFGTrainer`
  → `slat_denoiser_ema0.9999_step0075000.pt`

---

## 1. Installation

```bash
# pip
pip install -e .[train]
# CUDA extensions (see README): spconv, and flash-attn for exact reproduction
pip install spconv-cu120            # CUDA 12.x (or spconv-cu118)
pip install flash-attn --no-build-isolation

# or pixi
pixi install -e train             # CUDA 12.1 (use -e cu118-train for CUDA 11.8)
pixi run post-install               # flash-attn
```

Docker: the shipped `Dockerfile` includes the training dependencies.

> **flash-attn matters for reproduction.** The released checkpoints were
> trained with `ATTN_BACKEND=flash_attn`. Without flash-attn the code falls
> back to xformers/SDPA — training works, but does not bit-reproduce the
> original numerics. `train.py` prints a warning when the backend differs.

## 2. Checkpoints

```bash
python scripts/download_pretrained.py            # → ./checkpoints/RecGen + ./checkpoints/TRELLIS-image-large
python scripts/download_pretrained.py --with-dinov2   # also pre-cache DINOv2 for offline clusters
```

The image encoder (DINOv2 `dinov2_vitl14_reg`) is downloaded lazily via
`torch.hub` on the first training step; pre-seed `TORCH_HOME` on air-gapped
machines.

## 3. Quickstart on toy data

Generate a tiny deterministic WebDataset shard and run each stage end-to-end:

```bash
python scripts/make_toy_shards.py --out ./data/toy_wds --num-samples 32 --seed 0

# stage 1 (sparse structure), single GPU, 20 steps
python -m recgen_training.train \
    --config recgen_training/configs/toy/ss_stereo_flow_toy.json \
    --output_dir outputs/toy_ss --data_dir ./data --num_gpus 1

# stage 2 (SLAT)
python -m recgen_training.train \
    --config recgen_training/configs/toy/slat_flow_toy.json \
    --output_dir outputs/toy_slat --data_dir ./data --num_gpus 1

# multi-GPU smoke (DDP)
python -m recgen_training.train \
    --config recgen_training/configs/toy/ss_stereo_flow_toy.json \
    --output_dir outputs/toy_ss_ddp --data_dir ./data --num_gpus 2

# resume (picks up the latest ckpts/misc_step*.pt in output_dir)
python -m recgen_training.train \
    --config recgen_training/configs/toy/ss_stereo_flow_toy.json \
    --output_dir outputs/toy_ss --data_dir ./data --num_gpus 1 --ckpt latest
```

## 4. Data format

Training reads local [WebDataset](https://github.com/webdataset/webdataset)
tar shards. The `shards` config value is a brace-expansion pattern over one or
more dataset directories; `{data_dir}` is substituted with `--data_dir`:

```
{data_dir}/my_wds/{DatasetA_wds/shard-{000000..000123},DatasetB_wds/shard-{000000..000045}}.tar
```

Each sample (one object) carries these members (`<key>.<field>`):

| Member | Description |
|---|---|
| `sha256.txt` | object id, 64-hex (used by blacklists) |
| `ss_latent.npy` | stage-1 latent `float32 (8,16,16,16)` (TRELLIS SS-VAE) |
| `slat_coords.npy` | stage-2 sparse coords `int (N,3)` in `[0,64)`, `N ≤ 32768` |
| `slat_feats.npy` | stage-2 sparse features `float32 (N,8)` |
| `cond_image_{v:02d}.jpg` | per-view RGB, uint8 (must be `.jpg`) |
| `cond_mask_{v:02d}.png` | object mask, uint8 (a view needs **> 400** px) |
| `cond_mask_full_{v:02d}.png` | full-object mask (falls back to `cond_mask`) |
| `cond_depth_{v:02d}.png` | depth in **millimeters**, uint16 (0 = invalid) |
| `cond_depth_st_{v:02d}.png` | optional stereo depth (enables `mix_stereo_depth`) |
| `num_views.json` | number of stored views |
| `view_indices.json` | original render indices |
| `pose_data_<variant>.json` | per-view pose (keys below; fallback `pose_data.json`) |
| `pose_data_st_<variant>.json` | pose from stereo depth (for `mix_stereo_depth`) |
| `view_metadata.json` | per-view `{intrinsics 3×3, visible_fraction}` |
| `aesthetic_score.json` | SLAT training filters at ≥ 4.5 |
| `dataset_type.txt` | `objectbased` or `partbased` (per-type pose logging) |
| `captions.txt` | optional |
| `cond_mask_sam2_{v:02d}.png` | optional SAM2 mask (enables `mask_source: sam2`/`mix`) |
| `pose_data_sam2_<variant>.json` | pose in the SAM2 mask frame (required with SAM2 masks) |

Pose dicts carry `quat_{w,x,y,z}`, `trans_{x,y,z}`, `scale`, `rot6d_0..5`,
`rot9d_0..8`. Per-view quality filters (a sample needs ≥ `num_views` surviving
views): `visible_fraction ≥ 0.2`, mask px > 400, `0.3 < scale < 3.0`,
`‖translation‖ < 4`, `‖[quat, trans, scale]‖ < 10`.

> **Shard count vs GPUs**: WebDataset splits shards across ranks. The total
> number of shards must be ≥ the total number of training ranks — a rank
> that receives zero shards deadlocks DDP at the first gradient sync.

The paper's pose variant is `median_quantile_5per` (SAM3D-style robust
normalization of the depth pointmap; 5% quantile). `scripts/make_toy_shards.py`
is a working reference implementation of the schema.

**Optional augmentations** (dataset config keys):

* `color_jitter` (stage 1 only): torchvision ColorJitter kwargs, e.g.
  `{"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}` — one
  photometric transform sampled per object, applied to all its views. Rejected
  for SLAT training (the SLAT latents encode appearance, so jittering the
  conditioning breaks input/target color consistency).
* `mask_source`: `"gt"` (default) | `"sam2"` | `"mix"` — condition on
  SAM2-predicted masks instead of (or mixed 50/50 with, `sam2_mask_prob`)
  ground-truth masks, to robustify against segmentation noise at inference.
  Per view, the SAM2 mask is used only when its normalization frame is close
  to the GT frame (`sam2_max_log2_scale`, `sam2_max_dt`); otherwise that view
  falls back to GT mask + GT-frame pose.

**Pose normalization stats.** Set `pose_stats_file` in the dataset config to a
pre-aggregated stats JSON (recommended — this is what the shipped configs do),
or place `pose_stats_<variant>.json` / `pose_stats_st_<variant>.json` files in
each dataset directory (or its `additional_metadata/`) and the loader will
aggregate them, weighting stereo/normal stats by stereo availability. Generate
per-dataset stats for your own data with `scripts/compute_pose_stats.py`.

## 5. Reproducing the paper training

The training renderings (conditioning images, depth, masks, poses, and
precomputed latents) are publicly released at
`https://tri-ml-public.s3.amazonaws.com/github/recgen/train/` — see the
[dataset README](https://tri-ml-public.s3.amazonaws.com/github/recgen/train/README.md)
for licensing and attribution. The data is split into two collections
matching the two training stages:

* `ss/`   — stage-1 (sparse structure): `ss_latent.npy` targets, ~5 views.
* `slat/` — stage-2 (structured latent): `slat_coords.npy` + `slat_feats.npy`
  targets, ~20 views (aesthetic-filtered subset).

Download with:

```bash
# collection: {ss|slat|all}   subset: {ABO|HSSD|Objaverse|PartNeXt|all}
./scripts/download_data.sh /path/to/data/train_wds all all
```

The script downloads the gzipped shards, decompresses them, and renumbers
each collection/subset contiguously (recording the original shard index of
every file in `<collection>/<subset>/shard_map.json`). Please download once
and train from your own copy — do not stream shards from the release bucket.
After downloading, point each config's `shards` value at the released layout:

```
# stage 1 (ss_stereo_flow.json)
{data_dir}/train_wds/ss/{ABO/shard-{000000..000346},HSSD/shard-{000000..000519},Objaverse/shard-{000000..015619},PartNeXt/shard-{000000..001285}}.tar
# stage 2 (slat_flow.json)
{data_dir}/train_wds/slat/{ABO/shard-{000000..000398},HSSD/shard-{000000..000559},Objaverse/shard-{000000..014725},PartNeXt/shard-{000000..001353}}.tar
```

The shipped reproduce and finetune configs reference the TRI-internal shard
layout, so
substitute the patterns above. The public release is the redistributable
subset of the paper's training data (it excludes source datasets that do not
permit redistribution — PartNet-Mobility, PhysX-3D — and non-permissive
Objaverse/PartNeXt objects), so expect small deviations from the released
checkpoints when retraining on it.

The shipped configs mirror the exact run configs of the released checkpoints:

```bash
# stage 1: stereo sparse-structure flow (released ckpt = EMA 0.9999 @ step 55000)
python -m recgen_training.train \
    --config recgen_training/configs/reproduce/ss_stereo_flow.json \
    --output_dir outputs/ss_stereo_flow --data_dir /path/to/data \
    --num_gpus 8 --use_wandb

# stage 2: SLAT flow (released ckpt = EMA 0.9999 @ step 75000)
python -m recgen_training.train \
    --config recgen_training/configs/reproduce/slat_flow.json \
    --output_dir outputs/slat_flow --data_dir /path/to/data \
    --num_gpus 8 --use_wandb
```

Multi-node (paper scale: 8 nodes × 8 GPUs = 64 GPUs, global batch
8/GPU × 64 × batch_split 2):

```bash
# on each node i of N:
python -m recgen_training.train --config ... --output_dir ... --data_dir ... \
    --num_nodes N --node_rank i --num_gpus 8 \
    --master_addr <node0-address> --master_port 12356
# long snapshots/saves can exceed the NCCL watchdog — export NCCL_TIMEOUT=7200
```

Key hyperparameters (identical in both stages): AdamW lr 1e-4, EMA 0.9999,
fp16, logitNormal(1.0, 1.0) t-schedule, CFG p_uncond 0.1, view-dropout 0.33,
encoder `dinov2_vitl14_reg`; stage 1 adds pose_alpha 0.01. The run configs hold
the exact values (grad clipping, loss-scale growth, σ_min, …).

### Exactness notes

* **Backends & data order**: exact reproduction needs `ATTN_BACKEND=flash_attn`,
  `SPARSE_BACKEND=spconv`, numpy < 2, kornia < 0.8 (originals used torch 2.8 /
  CUDA 12.9). Sample order — and thus per-step batches — matches the originals
  only on the same 8-node × 8-GPU layout; loss curves are equivalent regardless.
* **Pose stats & blacklist**: the configs load the *training-time* pose stats in
  `configs/pose_stats/` — use these, not the inference-checkpoint stats. Both
  runs auto-load each dataset's `additional_metadata/blacklist_alignment.txt`;
  stage 1 also excludes a root blacklist, so repoint its `blacklist_file` at
  `{data_dir}/train_wds/ss/blacklist.txt` (stage 2 uses none).

## 6. Fine-tuning from the released checkpoints

```bash
python -m recgen_training.train \
    --config recgen_training/configs/finetune/ss_stereo_flow_from_recgen.json \
    --output_dir outputs/ft_ss --data_dir /path/to/your/data --num_gpus 8
```

The finetune configs (`finetune/ss_stereo_flow_from_recgen.json`,
`finetune/slat_flow_from_recgen.json`) are identical to the reproduce configs except
`finetune_ckpt` points at the released RecGen checkpoints (EMA weights),
`max_steps` 20000 and `i_save` 1000. Point `shards` at your own data (Section
4) and consider lowering the learning rate for small datasets: in our
small-data checks (a few thousand objects and below), 1e-5–3e-5 adapted more
smoothly than the paper's 1e-4, which also works but is noisier. On 24 GB
GPUs, set `"use_checkpoint": true` in the model args and use
micro-batch 1 (`batch_size_per_gpu` = `batch_split`).

The released checkpoints are EMA weights only — no optimizer state ships with
them, so `--ckpt`-style resume applies to *your own* runs (which write
`misc_step*.pt`); loading the released checkpoints always goes through
`finetune_ckpt`. (The trainer's resume format is fully compatible with the
original training runs: given a run directory with raw + EMA + `misc`
checkpoints, `--load_dir <dir> --ckpt <step>` restores optimizer, loss-scale,
and grad-clipper state and continues training.)

`finetune_from` loads with `strict=False`: keys missing from the checkpoint
(none, when starting from a released RecGen checkpoint; the new conditioning
embedders, when starting from a raw TRELLIS base) keep their fresh
initialization, and shape-mismatched keys are skipped with a warning. Note
that EMA parameters start from the *initialized* model, not the loaded
checkpoint — this matches the original training exactly.

## 7. Checkpoint format and inference handoff

Every `i_save` steps the trainer writes to `<output_dir>/ckpts/`:

* `denoiser_step{step:07d}.pt` — raw training weights (fp32 master params)
* `denoiser_ema{rate}_step{step:07d}.pt` — EMA weights (what we release)
* `misc_step{step:07d}.pt` — optimizer / log_scale / grad-clipper state
  (needed only for resuming; contains no RNG state, so a resumed run is
  statistically equivalent but not bit-identical to an uninterrupted one)

To run inference with your own checkpoint, assemble a directory shaped like
the released one and point the pipeline at it:

```python
from recgen_inference import build_recgen, generate
pipe = build_recgen.build(
    checkpoint_sparse="my_ckpts/stereo/denoiser_ema0.9999_step0020000.pt",
    checkpoint_slat="my_ckpts/slat/denoiser_ema0.9999_step0020000.pt",
)
```

(Place the matching `stereo_config.json` / `slat_config.json` and pose stats
next to the weights, or rely on the HuggingFace defaults — see
`recgen_inference/build_recgen.py`.)
