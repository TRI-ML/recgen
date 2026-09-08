# RecGen

This repository contains inference and training code for **RecGen**, a model for single-view and multi-view 3D reconstruction from RGB-D.
Given an RGB image, a depth map, an object mask, and camera intrinsics, RecGen produces a textured mesh, a Gaussian splat, and the object's 6-DoF pose in the camera frame.

## Setup

RecGen depends on `torch`, a few 3D libraries, and two CUDA extensions (`spconv`, `diff-gaussian-rasterization`). We recommend [pixi](https://pixi.sh), which pins Python, CUDA, and PyTorch in a single lockfile:

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # one-time
pixi install                                    # CUDA 12.1
pixi install -e cu118                           # CUDA 11.8
```

Optional for speedups and additional visualizations (after `pixi install`):

```bash
pixi run post-install                 # flash-attn (falls back to xformers if the build fails)
pixi run build-nvdiffrast             # nvdiffrast, needed for mp4 vis of the rendering as well as glb (--save-glb)
pixi run build-gaussian-rasterizer    # diff_gaussian_rasterization, needed for turntable.mp4 and --save-glb
```

Without these, `mesh.obj`, `overlay.png`, and `metadata.json` are still produced; the turntable video is skipped.

<details>
<summary>Manual install (no pixi)</summary>

```bash
pip install -e .
bash scripts/setup_cuda.sh            # spconv + flash-attn + diff-gaussian-rasterization
bash scripts/setup_cuda.sh --all      # also builds nvdiffrast
```

You are responsible for providing a working PyTorch + CUDA toolchain.
</details>

<details>
<summary>Docker (no local install)</summary>

If you'd rather not install anything on the host, a CUDA 12.1 image is provided. Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker build -t recgen-inference .
docker run --rm --gpus all -v $PWD/data:/data recgen-inference
```

Or via Compose (persists the HuggingFace cache in a named volume):

```bash
docker compose run --rm recgen bash
```

</details>

## Quick Start

```python
from recgen_inference import build_recgen, generate

pipeline = build_recgen.build("recgen_base.multiview_stereo")

result = generate(
    pipeline,
    image=rgb,           # HxWx3 uint8
    depth=depth,         # HxW uint16 (mm) or float32 (m) 
    mask=mask,           # HxW uint8, non-zero = object
    intrinsics=K,        # 3x3
)

print(result.pose_matrix)           # (4, 4) float64 pose in the camera frame
print(result.pose_quat)              # (7,) [tx, ty, tz, qx, qy, qz, qw]
print(result.mesh.vertices.shape)    # final textured mesh

result.save("./out", save_splat=True, save_glb=False)
```

`result` is a [`RecGenResult`](recgen_inference/_result.py) dataclass that carries the final mesh, the raw object-frame mesh, the 6-DoF pose (both as a 4×4 matrix and a 7-vector).

See [notebooks/example.ipynb](notebooks/example.ipynb) for a runnable walkthrough, or run the CLI script:

```bash
pixi run python scripts/run_inference.py \
    --rgb examples/ex0_rgb.png \
    --depth examples/ex0_depth.png \
    --mask examples/ex0_mask.png \
    --intrinsics examples/intrinsics.yaml \
    --out ./out --save-splat
```

`intrinsics.yaml` contains `fu`, `fv`, `pu`, `pv` as top-level keys.

## Multi-view Inference

```python
from recgen_inference import generate_multiview

result = generate_multiview(
    pipeline,
    anchor_view={"rgb": rgb0, "depth": d0, "mask": m0, "camera_intrinsics": K},
    second_views=[
        {"rgb": rgb1, "depth": d1, "mask": m1, "camera_intrinsics": K},
    ],
)
```

The result is expressed in the anchor view's camera frame. Any number of
second views is supported: the model is natively 2-view, so with N > 1 second
views the (anchor, view_i) conditioning pairs are fused at every denoising
step. More views improve fine detail and far-side consistency.

```python
result = generate_multiview(
    pipeline,
    anchor_view=view0,
    second_views=[view1, view2, view3],       # any N >= 1
    multidiffusion_mode="multidiffusion",     # or "stochastic"
    slat_multiview_mode=None,                 # "multidiffusion" to also fuse appearance
)
```

- `multidiffusion` (default) — averages the velocity predictions over all
  pairs at every step. Best quality; runs the model once per pair per step.
- `stochastic` — uses one pair per step, round-robin. No extra compute;
  slightly lower quality. Prefer for latency-sensitive use.

`slat_multiview_mode` optionally applies the same fusion to the appearance
(SLAT) stage; by default it uses the anchor + first second view. Best-quality
setting: `multidiffusion_mode="multidiffusion", slat_multiview_mode="multidiffusion"`.

## Post-hoc color calibration

`generate(..., posthoc_color="gamma_affine")` (CLI: `--posthoc-color gamma_affine`)
fits a gamma + per-channel-affine color transform between the generated asset
(rendered into the input view) and the input photo, and applies it to the
Gaussian and mesh colors so the asset matches the photo's exposure and color
balance. If the fit fails its quality gates the asset is left unchanged. The
fitted transform and statistics are stored in `result.posthoc_color` and
written to `posthoc_color.json`. Variants: `gamma`, `affine`, `gamma_affine`.

## Input Format

- **RGB** — any common format readable by OpenCV (PNG/JPG).
- **Depth** — 16-bit PNG in millimeters *or* float32 in meters. The unit is auto-detected.
- **Mask** — single-channel PNG; non-zero pixels mark the object.
- **Intrinsics** — `(3, 3)` matrix. Helper: `recgen_inference.utils.intrinsics_from_params(fx, fy, cx, cy)`.

## Output Files (`result.save(...)`)

| File | Description |
| --- | --- |
| `mesh.obj` | mesh in object frame (with vertex colors baked into MTL) |
| `posed_mesh.obj` | mesh in camera frame (with vertex colors baked into MTL) |
| `overlay.png` | posed mesh rendered over the input RGB (purple, software rasterizer) |
| `turntable.mp4` | 4 s side-by-side turntable: gaussian color \| mesh normals (skipped with a warning if nvdiffrast + diff_gaussian_rasterization are not installed) |
| `gaussian.ply` | Gaussian splat in object frame (if `save_splat=True`) |
| `posed_gaussian.ply` | Gaussian splat in camera frame (if `save_splat=True`) |
| `textured_mesh.glb` | mesh with baked texture in object frame (if `save_glb=True` and nvdiffrast + diff_gaussian_rasterization are installed) |
| `metadata.json` | predicted pose + camera info |

## Viewing Gaussian Splats

The `gaussian.ply` produced with `--save-splat` is compatible with [SuperSplat](https://superspl.at/editor), a browser-based viewer and editor. Drag the file into the editor window — no upload or install required, everything runs locally in the browser. SuperSplat is also useful for cropping, cleaning, and re-exporting splats (`.ply`, `.splat`, or compressed `.ply`).

## Troubleshooting

- **`spconv` import error** — wrong CUDA variant. Reinstall with `pip install spconv-cu118` or `spconv-cu120` to match your CUDA.
- **`flash-attn` build fails** — safe to ignore; RecGen falls back to PyTorch SDPA.
- **`diff_gaussian_rasterization` error on CUDA 12+** — rebuild from source via `bash scripts/setup_cuda.sh`; the prebuilt wheel targets CUDA 11.
- **GLB export fails** — run `pixi run build-nvdiffrast` and `pixi run build-gaussian-rasterizer`, or set `save_glb=False`.
- **OpenGL / EGL errors on headless servers** — `PYOPENGL_PLATFORM=egl` is set automatically; ensure your driver has EGL support.

## Dataset

The multi-view training renderings behind both released checkpoints
(conditioning images, depth, masks, poses, and precomputed latents for the
ABO, HSSD, Objaverse, and PartNeXt subsets) are publicly released as
WebDataset shards, split into two collections matching the two training
stages:

* `ss/` — stage-1 (sparse structure) collection, ~5 views per object.
* `slat/` — stage-2 (structured latent) collection, ~20 views per object
  (aesthetic-filtered subset).

```bash
# collection: {ss|slat|all}   subset: {ABO|HSSD|Objaverse|PartNeXt|all}
./scripts/download_data.sh /path/to/data/train_wds all all
```

See the [dataset README](https://tri-ml-public.s3.amazonaws.com/github/recgen/train/README.md)
for licensing and attribution, and [TRAINING.md](TRAINING.md) §4–5 for the
sample schema and how the shipped configs consume the released layout.

The evaluation benchmarks (HomebrewedDB and ArtVIP) are downloaded separately
via `scripts/download_eval_data.py` — see [Evaluation](#evaluation) below.

## Training & Fine-tuning

The `recgen_training` package can reproduce the training of both released
checkpoints (fine-tuned from [TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large))
and fine-tune from the released RecGen checkpoints on your own data:

```bash
pip install -e .[train]                     # or: pixi install -e train
python scripts/download_pretrained.py       # released + TRELLIS base checkpoints
python scripts/make_toy_shards.py --out ./data/toy_wds   # tiny smoke-test dataset
python -m recgen_training.train \
    --config recgen_training/configs/toy/ss_stereo_flow_toy.json \
    --output_dir outputs/toy_ss --data_dir ./data --num_gpus 1
```

See [TRAINING.md](TRAINING.md) for the data format, paper-reproduction
commands, multi-node launch, and the fine-tuning guide.

## Evaluation

The `recgen_eval` package reproduces the paper's shape/pose evaluation on the
HomebrewedDB (HB) and ArtVIP benchmarks with the released checkpoints:

```bash
pixi install -e eval && pixi run build-nvdiffrast && pixi run build-gaussian-rasterizer
python scripts/download_eval_data.py --dataset hb --out ./data/eval
python -m recgen_eval.run --dataset_name HB --num_views 1 \
    --path_to_datasets ./data/eval --save_path_root outputs/eval/hb1 \
    --parallel 1 --workers_per_gpu 2 --visualize 0 --save_videos 0
```

See [EVALUATION.md](EVALUATION.md) for the full protocol, the expected numbers,
and the ArtVIP benchmark download.

## License

This project is released for non-commercial use only. See [LICENSE](LICENSE) for full terms.

## Citation

```bibtex
@misc{zadaianchuk2026recgen,
      title={Reconstruction by Generation: 3D Multi-Object Scene Reconstruction from Sparse Observations},
      author={Andrii Zadaianchuk and Leonardo Barcellona and Lennard Schuenemann and Christian Gumbsch and Zehao Wang and Muhammad Zubair Irshad and Fabien Despinoy and Rahaf Aljundi and Stratis Gavves and Sergey Zakharov},
      year={2026},
      url={https://arxiv.org/abs/2604.27106},
}
```
