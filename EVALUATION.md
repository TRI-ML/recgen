# RecGen Evaluation (HB + ArtVIP)

This repository ships `recgen_eval`, a standalone port of the evaluation used
for the RecGen paper's main quantitative table on the **HomebrewedDB (HB)** and
**ArtVIP** benchmarks. It reproduces, with the released checkpoints, the
following published RecGen rows (mean over 538 HB / 500 ArtVIP instances):

| Benchmark | Views | CD_norm ↓ | ADD-SB ↓ | ADD-SB@0.1 ↑ | ADD-SB@0.05 ↑ | DRE@0.05 ↑ |
|---|---|---|---|---|---|---|
| HB     | 1 | 0.032 | 0.049 | 95.0 % | 73.8 % | 51.5 % |
| HB     | 2 | 0.029 | 0.048 | 95.4 % | 74.2 % | 50.9 % |
| ArtVIP | 1 | 0.026 | 0.034 | 96.4 % | 84.0 % | 24.4 % |
| ArtVIP | 2 | 0.024 | 0.032 | 96.4 % | 86.4 % | 24.8 % |

Metric definitions: **ADD-SB** = bidirectional (symmetric) Chamfer distance
between the generated mesh and the GT mesh, without ICP, normalized by the GT
diameter; **@0.1/@0.05** = recall at 10 %/5 % of the diameter; **CD_norm** =
diameter-normalized Chamfer after an additional ICP refinement; **DRE@0.05** =
fraction of samples whose predicted mesh diameter is within 5 % of the GT
diameter. The predicted pose is the generated mesh's centroid in the first
view's camera frame (identity rotation).

## 1. Install

```bash
git clone https://github.com/TRI-ML/recgen && cd recgen
curl -fsSL https://pixi.sh/install.sh | bash
pixi install -e eval                # CUDA 12.1 (use -e cu118-eval for CUDA 11.8)
pixi run post-install               # flash-attn (optional; xformers/SDPA fallback)
pixi run build-nvdiffrast           # required: GLB mesh post-processing
pixi run build-gaussian-rasterizer  # required: GLB texture baking
pixi shell -e eval                  # activate (or prefix commands with `pixi run -e eval`)
```

pip alternative: `pip install -e .[glb,eval] && bash scripts/setup_cuda.sh --all`
plus `pip install spconv-cu120` (or `spconv-cu118`) matching your CUDA.

## 2. Checkpoints

The evaluation defaults to the released paper checkpoints on HuggingFace
([TRI-ML/RecGen](https://huggingface.co/TRI-ML/RecGen)) — the stereo
sparse-structure denoiser (EMA, step 55k) and the SLAT denoiser (EMA, step
75k), exactly the weights behind the table above. They are auto-downloaded on
first use; pass `--ckpt_path_structure/--ckpt_path_slats` for local files.

## 3. Data

```bash
# HB — official BOP benchmark distribution (~13 GB download, ~34 GB extracted)
python scripts/download_eval_data.py --dataset hb --out ./data/eval

# ArtVIP — RecGen eval benchmark (~36 GB), custom IsaacSim renders of ArtVIP assets
python scripts/download_eval_data.py --dataset artvip --out ./data/eval
```

HB comes from `bop-benchmark/hb` on HuggingFace (`hb_base.zip`,
`hb_models.zip`, `hb_val_kinect.zip`); please follow the BOP/HomebrewedDB
licenses and cite them. The fixed 538/500-sample evaluation splits used in the
paper ship inside this package (`recgen_eval/splits/`) and are picked up
automatically. The ArtVIP benchmark download is verified against a sha256
manifest.

## 4. Run

One command per table row (on a 4-GPU machine). Use **1 worker per GPU on
24 GB cards** — the mesh post-processing peak makes two pipelines per GPU OOM
on 23–24 GB; 2–3 workers/GPU need 40 GB+ GPUs. Any OOM-failed sample is
penalty-scored and invalidates the paper comparison (see §6):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduces fragmentation
python -m recgen_eval.run --dataset_name HB       --num_views 1 --path_to_datasets ./data/eval \
    --save_path_root outputs/eval/hb1 --parallel 1 --workers_per_gpu 1 --visualize 0 --save_videos 0
python -m recgen_eval.run --dataset_name HB       --num_views 2 --path_to_datasets ./data/eval \
    --save_path_root outputs/eval/hb2 --parallel 1 --workers_per_gpu 1 --visualize 0 --save_videos 0
python -m recgen_eval.run --dataset_name AV_final --num_views 1 --path_to_datasets ./data/eval \
    --save_path_root outputs/eval/artvip1 --parallel 1 --workers_per_gpu 1 --visualize 0 --save_videos 0
python -m recgen_eval.run --dataset_name AV_final --num_views 2 --path_to_datasets ./data/eval \
    --save_path_root outputs/eval/artvip2 --parallel 1 --workers_per_gpu 1 --visualize 0 --save_videos 0
```

Defaults match the paper protocol: seed 1, 25-step flow sampling for both
stages, packaged instance lists (HB 2-view uses the "simple" pairing, all
others the "random" one), mask erosion 3×3 (auto-disabled for ArtVIP), robust
median-quantile pointmap normalization (5 % tails), clamp [-2, 3]. Runs are
resumable: re-running with `--overwrite 0` skips finished samples
(per-sample `metrics.json`).

Interrupted or partially aggregated runs can be re-aggregated without
inference: `python -m recgen_eval.aggregate --run_dir <run_dir>`.

## 5. Results

Each run writes
`<save_path_root>/multiview_eval/pointmap_True_predicted_scale_True/dataset_<DS>_multiview<N>/`
(`<run_dir>` below): per-sample `obj_*/metrics.json` + mesh artifacts, and the
aggregate CSVs under `<run_dir>/predictions/final_results_from_json/` — the
MEAN row of `0_all_frames_metrics_results_statistics.csv` is the table number.

DRE@0.05 and the final side-by-side against the published values:

```bash
python -m recgen_eval.diameter --eval_dir <run_dir> --data_root ./data/eval --dataset HB
python -m recgen_eval.table    --run_dir <run_dir> --expect hb1   # hb1|hb2|artvip1|artvip2
```

## 6. Runtime and reproducibility

* Runtime: roughly 30–60 s/sample/worker (visualization off). With 4 GPUs × 1
  worker: ≈ 1.5–2.5 h per row.
* A failure count > 0 in the output (`recgen_eval.table` warns loudly) means
  samples were penalty-scored — usually GPU OOM from too many workers per GPU —
  and the run is **not** comparable to the paper (which had 0 failures). Delete
  the failed samples' `metrics.json` and re-run with `--overwrite 0` to resume.
* Diffusion sampling is seeded (`--seed 1`, the paper setting) but
  GPU-arch/attention-backend dependent (flash-attn vs xformers vs SDPA), so
  per-sample outputs are not bitwise reproducible across hardware. Aggregate
  numbers are stable: across sampling seeds 1–5 the paper runs' HB 1-view
  spread was CD_norm 0.031–0.032, ADD-SB 0.046–0.050, @0.1 92.8–95.5,
  @0.05 72.5–74.7. Expect your means within ~±0.002 CD_norm / ±0.004 ADD-SB /
  ±2.5 pp recall of the table.
* The Chamfer metrics draw 99,999 random surface samples per mesh (unseeded),
  adding ~1 % per-sample jitter; aggregates over 500+ samples are stable to
  <5e-4.
* A sample that fails mesh generation twice is scored with the paper's
  failure penalty (Chamfer = 20 cm, recall 0). The paper runs had 0 failures.

## 7. Scope

`recgen_eval` is the exact evaluation stack behind the paper table (BOP
loaders, multi-view instance pairing, the anchor metric suite, CSV
aggregation), wired to the public `recgen_inference` package for all model
inference and preprocessing. It covers the HB and ArtVIP benchmarks and the
metrics in the table above; `tests/eval/` contains loader and metric unit
tests (`pytest tests/eval`).
