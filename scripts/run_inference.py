"""Run RecGen single-view inference from the command line.

Example:
    python scripts/run_inference.py \
        --rgb examples/ex0_rgb.png \
        --depth examples/ex0_depth.png \
        --mask examples/ex0_mask.png \
        --intrinsics examples/intrinsics.yaml \
        --name ex0
"""

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import yaml

from recgen_inference import build_recgen, generate


def load_intrinsics(path: str) -> np.ndarray:
    """Load `fu`, `fv`, `pu`, `pv` from a YAML file and return a (3, 3) matrix."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    missing = [k for k in ("fu", "fv", "pu", "pv") if k not in data]
    if missing:
        raise ValueError(f"intrinsics YAML missing keys: {missing}")
    fu, fv, pu, pv = (float(data[k]) for k in ("fu", "fv", "pu", "pv"))
    return np.array([[fu, 0.0, pu], [0.0, fv, pv], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_inputs(rgb_path: str, depth_path: str, mask_path: str):
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(depth_path)

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return rgb, depth, mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RecGen single-view inference")
    p.add_argument("--rgb", required=True, help="Path to RGB image (PNG/JPG)")
    p.add_argument("--depth", required=True, help="Path to depth map (uint16 mm or float32 m)")
    p.add_argument("--mask", required=True, help="Path to object mask (non-zero = object)")
    p.add_argument(
        "--intrinsics",
        required=True,
        help="Path to a YAML file containing [fu, fv, pu, pv]",
    )

    p.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to outputs/inference_outputs/<name>.",
    )
    p.add_argument(
        "--name",
        default="run",
        help="Subfolder name under outputs/inference_outputs/ when --out is not given.",
    )
    p.add_argument("--checkpoint", default="recgen_base.multiview_stereo", help="RecGen checkpoint name")
    p.add_argument(
        "--checkpoint-slat", default=None,
        help="Local path to a SLAT denoiser checkpoint (overrides the HF download)",
    )
    p.add_argument(
        "--checkpoint-sparse", default=None,
        help="Local path to a sparse-structure denoiser checkpoint (overrides the HF download)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-splat", action="store_true", help="Also write Gaussian splat (.ply)")
    p.add_argument("--save-glb", action="store_true", help="Also write textured GLB (requires nvdiffrast)")
    p.add_argument(
        "--posthoc-color", default="none", choices=["none", "gamma", "affine", "gamma_affine"],
        help="Post-hoc asset color calibration against the input image "
             "(falls back to identity when the fit is unreliable)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rgb, depth, mask = load_inputs(args.rgb, args.depth, args.mask)
    K = load_intrinsics(args.intrinsics)

    print(f"[run_inference] Loading pipeline: {args.checkpoint}")
    pipeline = build_recgen.build(
        args.checkpoint,
        checkpoint_slat=args.checkpoint_slat,
        checkpoint_sparse=args.checkpoint_sparse,
    )

    print("[run_inference] Running inference...")
    result = generate(pipeline, image=rgb, depth=depth, mask=mask, intrinsics=K, seed=args.seed,
                      posthoc_color=args.posthoc_color)

    print(f"[run_inference] Mesh: {result.mesh.vertices.shape[0]} vertices, "
          f"{result.mesh.faces.shape[0]} faces")
    print(f"[run_inference] Pose:\n{result.pose_matrix}")

    out_dir = Path(args.out) if args.out else REPO_ROOT / "outputs" / "inference_outputs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    result.save(out_dir, save_splat=args.save_splat, save_glb=args.save_glb)
    print(f"[run_inference] Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
