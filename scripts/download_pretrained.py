#!/usr/bin/env python3
"""
Download the checkpoints needed for RecGen training and fine-tuning.

Fetches into ./checkpoints/ (relative paths match the shipped training configs):

  checkpoints/RecGen/
      stereo_denoiser_ema0.9999_step0055000.pt   stage-1 paper checkpoint
      sparse-structure-ft-70k/                   stage-1 default (SAM2/ColorJitter
                                                 fine-tune) + its config/pose stats
      slat_denoiser_ema0.9999_step0075000.pt     stage-2 checkpoint
      stereo_config.json, slat_config.json       model configs
      stereo_pose_stats.json, slat_pose_stats.json  (inference-side pose stats)

  checkpoints/TRELLIS-image-large/ckpts/
      ss_flow_img_dit_L_16l8_fp16.safetensors    TRELLIS base for stage-1 training
      slat_flow_img_dit_L_64l8p2_fp16.safetensors  TRELLIS base for stage-2 training
      slat_dec_gs_swin8_B_64l8gs32_fp16.{json,safetensors}  (optional, SLAT snapshots)

Usage:
    python scripts/download_pretrained.py [--out ./checkpoints] [--with-dinov2] [--skip-recgen] [--skip-trellis]
"""

import argparse
import os
import shutil

RECGEN_REPO = "TRI-ML/RecGen"
TRELLIS_REPO = "microsoft/TRELLIS-image-large"

RECGEN_FILES = [
    "stereo_denoiser_ema0.9999_step0055000.pt",
    "sparse-structure-ft-70k/stereo_denoiser_ema0.9999_step0070000.pt",
    "sparse-structure-ft-70k/stereo_config.json",
    "sparse-structure-ft-70k/stereo_pose_stats.json",
    "slat_denoiser_ema0.9999_step0075000.pt",
    "stereo_config.json",
    "slat_config.json",
    "stereo_pose_stats.json",
    "slat_pose_stats.json",
]

TRELLIS_FILES = [
    "ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors",
    "ckpts/ss_flow_img_dit_L_16l8_fp16.json",
    "ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors",
    "ckpts/slat_flow_img_dit_L_64l8p2_fp16.json",
    # Optional: SLAT Gaussian decoder used by the SLAT trainer's visual snapshots
    "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors",
    "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.json",
]


def fetch(repo_id, filename, dest_path, fallback_repo=None, fallback_filename=None):
    from huggingface_hub import hf_hub_download

    if os.path.exists(dest_path):
        print(f"  exists: {dest_path}")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        src = hf_hub_download(repo_id, filename)
    except Exception as e:
        if fallback_repo is None:
            raise
        print(f"  {repo_id}/{filename} unavailable ({e}); trying {fallback_repo}")
        src = hf_hub_download(fallback_repo, fallback_filename or filename)
    shutil.copy(src, dest_path)
    print(f"  fetched: {dest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="./checkpoints", help="Checkpoints root directory")
    parser.add_argument("--with-dinov2", action="store_true",
                        help="Also pre-download the DINOv2 image encoder via torch.hub (for offline clusters)")
    parser.add_argument("--skip-recgen", action="store_true", help="Skip released RecGen checkpoints")
    parser.add_argument("--skip-trellis", action="store_true", help="Skip TRELLIS base checkpoints")
    args = parser.parse_args()

    if not args.skip_recgen:
        print(f"Downloading released RecGen checkpoints from {RECGEN_REPO}...")
        for f in RECGEN_FILES:
            fetch(RECGEN_REPO, f, os.path.join(args.out, "RecGen", f))

    if not args.skip_trellis:
        print(f"Downloading TRELLIS base checkpoints from {TRELLIS_REPO}...")
        for f in TRELLIS_FILES:
            # TRI-ML/RecGen mirrors the TRELLIS base files (self-contained repo)
            fetch(TRELLIS_REPO, f, os.path.join(args.out, "TRELLIS-image-large", f),
                  fallback_repo=RECGEN_REPO, fallback_filename=os.path.basename(f))

    if args.with_dinov2:
        print("Pre-downloading DINOv2 (dinov2_vitl14_reg) via torch.hub...")
        import torch
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True)
        print("  done (cached under TORCH_HOME).")

    print("\nAll requested checkpoints are in place.")


if __name__ == "__main__":
    main()
