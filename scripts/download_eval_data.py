#!/usr/bin/env python3
"""Download the RecGen evaluation data (HB and/or ArtVIP) into a local root.

HB (HomebrewedDB) comes from the official BOP benchmark distribution on
HuggingFace (datasets/bop-benchmark/hb): hb_base.zip + hb_models.zip +
hb_val_kinect.zip (~13 GB download, ~34 GB extracted). The archives are
extracted into the per-archive layout the evaluation loader expects:

    <out>/HB/hb_base/hb/camera_kinect.json
    <out>/HB/hb_models/{models,models_eval}/...
    <out>/HB/hb_val_kinect/val_kinect/000001..000013/{rgb,depth,mask,mask_visib,scene_*.json}

ArtVIP (custom IsaacSim-rendered BOP-format benchmark, ~36 GB) comes from the
RecGen eval-data HuggingFace dataset repo and is verified against the shipped
sha256 manifest.

Usage:
    python scripts/download_eval_data.py --dataset hb --out ./data/eval
    python scripts/download_eval_data.py --dataset artvip --out ./data/eval
    python scripts/download_eval_data.py --dataset all --out ./data/eval
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile

BOP_HB_REPO = "bop-benchmark/hb"
# (archive filename, target subdirectory under <out>/HB/)
HB_ARCHIVES = [
    ("hb_base.zip", "hb_base"),
    ("hb_models.zip", "hb_models"),
    ("hb_val_kinect.zip", "hb_val_kinect"),
]

ARTVIP_EVAL_REPO = os.environ.get("RECGEN_ARTVIP_EVAL_REPO", "TRI-ML/RecGen-ArtVIP-Eval")


def _hf_download(repo_id: str, filename: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=local_dir)


def download_hb(out_root: str, keep_archives: bool = False) -> None:
    hb_root = os.path.join(out_root, "HB")
    os.makedirs(hb_root, exist_ok=True)
    archive_dir = os.path.join(hb_root, "_archives")

    for archive, subdir in HB_ARCHIVES:
        target = os.path.join(hb_root, subdir)
        if os.path.isdir(target) and os.listdir(target):
            print(f"[hb] {target} already exists — skipping {archive}")
            continue
        print(f"[hb] downloading {archive} from {BOP_HB_REPO} ...")
        zip_path = _hf_download(BOP_HB_REPO, archive, archive_dir)
        print(f"[hb] extracting {archive} -> {target}")
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        if not keep_archives:
            os.remove(zip_path)

    if not keep_archives and os.path.isdir(archive_dir):
        shutil.rmtree(archive_dir, ignore_errors=True)

    validate_hb(hb_root)


def validate_hb(hb_root: str) -> None:
    """Cheap structural validation of the extracted HB tree."""
    cam = os.path.join(hb_root, "hb_base", "hb", "camera_kinect.json")
    assert os.path.exists(cam), f"missing {cam}"
    with open(cam) as f:
        cam_data = json.load(f)
    for key in ("fx", "fy", "cx", "cy", "width", "height", "depth_scale"):
        assert key in cam_data, f"camera_kinect.json missing key {key}"

    models_eval = os.path.join(hb_root, "hb_models", "models_eval")
    plys = [f for f in os.listdir(models_eval) if f.endswith(".ply")]
    assert len(plys) == 33, f"expected 33 models_eval PLYs, found {len(plys)}"
    assert os.path.exists(os.path.join(models_eval, "models_info.json"))

    scenes_dir = os.path.join(hb_root, "hb_val_kinect", "val_kinect")
    scenes = sorted(d for d in os.listdir(scenes_dir)
                    if os.path.isdir(os.path.join(scenes_dir, d)))
    assert len(scenes) == 13, f"expected 13 val_kinect scenes, found {len(scenes)}"
    for seq in scenes:
        for req in ("rgb", "depth", "mask_visib", "scene_gt.json",
                    "scene_gt_info.json", "scene_camera.json"):
            assert os.path.exists(os.path.join(scenes_dir, seq, req)), \
                f"scene {seq} missing {req}"
    print(f"[hb] validation OK: 13 scenes, 33 eval models, camera_kinect.json present")


def download_artvip(out_root: str) -> None:
    from huggingface_hub import snapshot_download

    av_root = os.path.join(out_root, "AV_final")
    print(f"[artvip] downloading {ARTVIP_EVAL_REPO} -> {av_root} ...")
    try:
        snapshot_download(ARTVIP_EVAL_REPO, repo_type="dataset", local_dir=av_root)
    except Exception as e:
        print(f"[artvip] ERROR: could not download {ARTVIP_EVAL_REPO}: {e}")
        print("[artvip] If the repo is gated/not yet public, request access on its "
              "HuggingFace page or set RECGEN_ARTVIP_EVAL_REPO to a mirror.")
        sys.exit(1)

    validate_artvip(av_root)


def validate_artvip(av_root: str, verify_hashes: bool = True) -> None:
    """Structural validation + (optional) sha256 manifest verification."""
    cam = os.path.join(av_root, "artvip_base", "artvip", "camera.json")
    assert os.path.exists(cam), f"missing {cam}"
    assert os.path.exists(os.path.join(av_root, "artvip_base", "artvip", "test_targets_all.json"))

    models_eval = os.path.join(av_root, "artvip_models", "models_eval")
    plys = [f for f in os.listdir(models_eval) if f.endswith(".ply")]
    assert len(plys) == 580, f"expected 580 models_eval PLYs, found {len(plys)}"
    assert os.path.exists(os.path.join(models_eval, "models_info.json"))

    scenes_dir = os.path.join(av_root, "artvip_test_all", "test")
    scenes = [d for d in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, d))]
    assert len(scenes) == 580, f"expected 580 test scene dirs, found {len(scenes)}"

    split = os.path.join(av_root, "fixed_split", "instance_list_random.txt")
    assert os.path.exists(split), f"missing {split}"
    with open(split) as f:
        n_lines = sum(1 for line in f if line.strip())
    assert n_lines == 500, f"expected 500 instance-list lines, found {n_lines}"

    manifest_path = os.path.join(av_root, "manifest_sha256.json")
    if verify_hashes and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"[artvip] verifying {len(manifest['files'])} file hashes ...")
        bad = 0
        for entry in manifest["files"]:
            p = os.path.join(av_root, entry["path"])
            if not os.path.exists(p) or os.path.getsize(p) != entry["size"]:
                print(f"  MISSING/SIZE-MISMATCH: {entry['path']}")
                bad += 1
                continue
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != entry["sha256"]:
                print(f"  HASH MISMATCH: {entry['path']}")
                bad += 1
        assert bad == 0, f"{bad} files failed manifest verification"
        print("[artvip] manifest verification OK")
    print(f"[artvip] validation OK: 580 scenes, 580 eval models, 500-sample split present")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["hb", "artvip", "all"], required=True)
    parser.add_argument("--out", type=str, default="./data/eval",
                        help="Eval-data root (passed to recgen_eval.run as --path_to_datasets)")
    parser.add_argument("--keep-archives", action="store_true",
                        help="Keep the downloaded HB zip archives after extraction")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate an existing tree, no download")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.dataset in ("hb", "all"):
        if args.validate_only:
            validate_hb(os.path.join(args.out, "HB"))
        else:
            download_hb(args.out, keep_archives=args.keep_archives)
    if args.dataset in ("artvip", "all"):
        if args.validate_only:
            validate_artvip(os.path.join(args.out, "AV_final"))
        else:
            download_artvip(args.out)

    print("\nDone. Run evaluations with:")
    print(f"  python -m recgen_eval.run --path_to_datasets {args.out} ...")


if __name__ == "__main__":
    main()
