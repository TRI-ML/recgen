#!/usr/bin/env python3
"""
Compute predicted object diameter relative error vs GT diameter (DRE).

For each predicted mesh (final_mesh_recgen.obj) in the evaluation output,
computes its diameter (max pairwise distance) and reports the relative error
compared to the ground-truth diameter from BOP models_info.json.
DRE@0.05 (paper table) = fraction of samples with rel_error < 0.05.

The script auto-detects whether predicted meshes are in meters or millimeters
by comparing the first few samples against GT diameters.


Usage:
    python -m recgen_eval.diameter \
        --eval_dir /path/to/dataset_HB_multiview1 \
        --data_root /path/to/eval_data --dataset HB
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import ConvexHull


# models_info.json locations relative to the eval-data root
DATASET_MODELS_INFO = {
    "HB": "HB/hb_models/models_eval/models_info.json",
    "AV_final": "AV_final/artvip_models/models_eval/models_info.json",
}


def compute_mesh_diameter(mesh):
    """Compute mesh diameter as the max pairwise distance on the convex hull.

    Uses convex hull vertices for efficiency (the diameter is always between
    two convex hull vertices). Falls back to vertex sampling if hull fails.
    """
    try:
        hull = ConvexHull(mesh.vertices)
        pts = mesh.vertices[hull.vertices]
    except Exception:
        if len(mesh.vertices) > 10000:
            idx = np.random.choice(len(mesh.vertices), 10000, replace=False)
            pts = mesh.vertices[idx]
        else:
            pts = mesh.vertices

    if len(pts) > 5000:
        idx = np.random.choice(len(pts), 5000, replace=False)
        pts = pts[idx]

    # Max pairwise distance via broadcasting
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    return float(dists.max())


def load_gt_diameters_mm(models_info_path):
    """Load GT diameters from BOP models_info.json in millimeters (raw BOP units).

    Returns dict {object_id (int): diameter_mm (float)}.
    """
    with open(models_info_path, 'r') as f:
        info = json.load(f)

    diameters = {}
    for obj_id_str, obj_info in info.items():
        obj_id = int(obj_id_str)
        diameters[obj_id] = obj_info['diameter']  # mm
    return diameters


def parse_object_id(folder_name):
    """Parse object_id from folder name like obj_000004_multiview_... or obj_000004_anchor_..."""
    parts = folder_name.split('_')
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


def detect_dataset_from_path(eval_dir):
    """Try to detect dataset name from the eval_dir path."""
    path_lower = eval_dir.lower()
    for ds in ["AV_final", "HB"]:
        if ds.lower() in path_lower:
            return ds
    return None


def detect_mesh_units(pred_diameters, gt_diameters_mm):
    """Auto-detect whether predicted meshes are in meters or millimeters.

    Compares a few samples: if pred/gt_mm ratio is ~1, meshes are in mm.
    If ratio is ~0.001, meshes are in meters.

    Returns scale factor to convert predicted diameters to mm.
    """
    ratios = []
    for obj_id, pred_diam in pred_diameters.items():
        if obj_id in gt_diameters_mm:
            ratios.append(pred_diam / gt_diameters_mm[obj_id])

    if not ratios:
        return 1.0  # default: assume mm

    median_ratio = np.median(ratios)

    # If ratio is ~0.001 (predicted is in meters, GT in mm)
    if median_ratio < 0.01:
        print(f"  Auto-detected mesh units: METERS (median ratio to GT_mm: {median_ratio:.6f})")
        return 1000.0  # multiply predicted by 1000 to get mm
    # If ratio is ~1 (both in mm)
    elif 0.1 < median_ratio < 10.0:
        print(f"  Auto-detected mesh units: MILLIMETERS (median ratio to GT_mm: {median_ratio:.4f})")
        return 1.0
    else:
        print(f"  WARNING: Unexpected diameter ratio {median_ratio:.4f}. Assuming mm.")
        return 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Evaluation output directory containing obj_* folders with final_mesh_recgen.obj")
    parser.add_argument("--models_info", type=str, default=None,
                        help="Path to BOP models_info.json (auto-detected if --dataset and --data_root are given)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Eval-data root containing HB/ and/or AV_final/ (used with --dataset)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (HB, AV_final) for auto-detecting models_info under --data_root")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV path (default: <eval_dir>/diameter_error.csv)")
    parser.add_argument("--mesh_name", type=str, default="final_mesh_recgen.obj",
                        help="Name of predicted mesh file to load")
    parser.add_argument("--units", type=str, default="auto", choices=["auto", "mm", "m"],
                        help="Units of predicted meshes (default: auto-detect)")
    parser.add_argument("--filter_samples", type=str, default=None,
                        help="CSV file with sample names to filter to (column 4 = folder name, no header)")
    args = parser.parse_args()

    # Load filter set if provided
    filter_folders = None
    if args.filter_samples:
        filter_df = pd.read_csv(args.filter_samples, header=None,
                                names=["obj_id", "seq_id", "img_id", "sample_name"])
        filter_folders = set(filter_df["sample_name"].str.strip())
        print(f"Filtering to {len(filter_folders)} samples from {args.filter_samples}")

    # Resolve models_info path
    if args.models_info is None:
        dataset = args.dataset or detect_dataset_from_path(args.eval_dir)
        if dataset is None:
            print("ERROR: Cannot auto-detect dataset. Specify --dataset or --models_info.")
            sys.exit(1)
        if dataset not in DATASET_MODELS_INFO:
            print(f"ERROR: Unknown dataset '{dataset}'. Known: {list(DATASET_MODELS_INFO.keys())}")
            sys.exit(1)
        if args.data_root is None:
            print("ERROR: --data_root is required when auto-detecting models_info from --dataset.")
            sys.exit(1)
        args.models_info = os.path.join(args.data_root, DATASET_MODELS_INFO[dataset])
        print(f"Auto-detected dataset: {dataset}")

    if not os.path.exists(args.models_info):
        print(f"ERROR: models_info not found: {args.models_info}")
        sys.exit(1)

    # Load GT diameters in mm (raw BOP units)
    gt_diameters_mm = load_gt_diameters_mm(args.models_info)
    print(f"Loaded GT diameters for {len(gt_diameters_mm)} objects")

    # Find all predicted meshes (mesh_name can be a glob pattern)
    mesh_files = sorted(glob.glob(os.path.join(args.eval_dir, "obj_*", args.mesh_name)))
    if not mesh_files:
        print(f"No {args.mesh_name} files found in {args.eval_dir}/obj_*/")
        sys.exit(1)
    # If glob matched multiple meshes per folder, keep only the first per folder
    seen_folders = set()
    unique_mesh_files = []
    for m in mesh_files:
        folder = os.path.dirname(m)
        if folder not in seen_folders:
            seen_folders.add(folder)
            unique_mesh_files.append(m)
    mesh_files = unique_mesh_files
    # Apply sample filter if provided
    if filter_folders is not None:
        mesh_files = [m for m in mesh_files
                      if os.path.basename(os.path.dirname(m)) in filter_folders]
    print(f"Found {len(mesh_files)} predicted meshes" +
          (f" (after filtering)" if filter_folders else ""))

    # First pass: compute all predicted diameters (raw)
    raw_diameters = {}  # {mesh_path: (obj_id, pred_diam_raw)}
    probe_by_obj = {}  # for unit detection: {obj_id: pred_diam}
    for mesh_path in mesh_files:
        obj_folder = os.path.basename(os.path.dirname(mesh_path))
        obj_id = parse_object_id(obj_folder)
        if obj_id is None or obj_id not in gt_diameters_mm:
            continue

        try:
            mesh = trimesh.load(mesh_path, force='mesh')
        except Exception:
            continue

        if len(mesh.vertices) < 4:
            continue

        pred_diam = compute_mesh_diameter(mesh)
        raw_diameters[mesh_path] = (obj_id, pred_diam)

        if obj_id not in probe_by_obj:
            probe_by_obj[obj_id] = pred_diam

    # Detect units
    if args.units == "auto":
        scale_to_mm = detect_mesh_units(probe_by_obj, gt_diameters_mm)
    elif args.units == "m":
        scale_to_mm = 1000.0
        print("  Mesh units: METERS (user-specified)")
    else:
        scale_to_mm = 1.0
        print("  Mesh units: MILLIMETERS (user-specified)")

    # Second pass: compute metrics with correct units
    rows = []
    errors_by_obj = {}
    for mesh_path in mesh_files:
        obj_folder = os.path.basename(os.path.dirname(mesh_path))
        obj_id = parse_object_id(obj_folder)
        if obj_id is None:
            print(f"  [WARN] Cannot parse object_id from: {obj_folder}")
            continue

        gt_diam_mm = gt_diameters_mm.get(obj_id)
        if gt_diam_mm is None:
            print(f"  [WARN] No GT diameter for object {obj_id}")
            continue

        if mesh_path not in raw_diameters:
            continue

        _, pred_diam_raw = raw_diameters[mesh_path]
        pred_diam_mm = pred_diam_raw * scale_to_mm
        rel_error = abs(pred_diam_mm - gt_diam_mm) / gt_diam_mm
        ratio = pred_diam_mm / gt_diam_mm
        # Score: 1 = perfect match, 0 = completely wrong
        score = min(ratio, 1.0 / ratio) if ratio > 0 else 0.0

        rows.append({
            'folder': obj_folder,
            'object_id': obj_id,
            'pred_diameter_mm': pred_diam_mm,
            'gt_diameter_mm': gt_diam_mm,
            'diameter_ratio': ratio,
            'rel_error': rel_error,
            'diameter_score': score,
        })

        if obj_id not in errors_by_obj:
            errors_by_obj[obj_id] = []
        errors_by_obj[obj_id].append(rel_error)

    # Add zero-score rows for missing samples (failed inference)
    if filter_folders is not None:
        found_folders = {r['folder'] for r in rows}
        missing = filter_folders - found_folders
        if missing:
            print(f"  {len(missing)} missing samples (failed inference) — assigned score=0")
            for folder_name in sorted(missing):
                obj_id = parse_object_id(folder_name)
                gt_diam_mm = gt_diameters_mm.get(obj_id) if obj_id else None
                rows.append({
                    'folder': folder_name,
                    'object_id': obj_id,
                    'pred_diameter_mm': 0.0,
                    'gt_diameter_mm': gt_diam_mm or 0.0,
                    'diameter_ratio': 0.0,
                    'rel_error': 1.0,
                    'diameter_score': 0.0,
                })
                if obj_id is not None:
                    if obj_id not in errors_by_obj:
                        errors_by_obj[obj_id] = []
                    errors_by_obj[obj_id].append(1.0)

    if not rows:
        print("No valid results.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Print per-object summary
    print(f"\n{'='*80}")
    print("Per-object summary (diameter relative error)")
    print(f"{'='*80}")
    print(f"{'Obj ID':>8s} {'N':>5s} {'GT(mm)':>8s} {'Pred(mm)':>8s} {'Score':>8s} {'Ratio':>8s} {'Err':>8s}")
    print("-" * 60)
    for obj_id in sorted(errors_by_obj.keys()):
        obj_rows = df[df['object_id'] == obj_id]
        gt_d = obj_rows['gt_diameter_mm'].iloc[0]
        n = len(obj_rows)
        mean_pred = obj_rows['pred_diameter_mm'].mean()
        mean_score = obj_rows['diameter_score'].mean()
        mean_ratio = obj_rows['diameter_ratio'].mean()
        mean_err = obj_rows['rel_error'].mean()
        print(f"{obj_id:>8d} {n:>5d} {gt_d:>8.1f} {mean_pred:>8.1f} {mean_score:>8.4f} {mean_ratio:>8.4f} {mean_err:>8.4f}")

    # Overall
    print("-" * 60)
    mean_err = df['rel_error'].mean()
    median_err = df['rel_error'].median()
    mean_ratio = df['diameter_ratio'].mean()
    mean_score = df['diameter_score'].mean()
    median_score = df['diameter_score'].median()
    print(f"{'ALL':>8s} {len(df):>5d} {'':>8s} {'':>8s} {mean_score:>8.4f} {mean_ratio:>8.4f} {mean_err:>8.4f}")
    print(f"\nOverall statistics:")
    print(f"  Mean diameter score:   {mean_score:.4f}")
    print(f"  Median diameter score: {median_score:.4f}")
    print(f"  Mean relative error:   {mean_err:.4f} ({mean_err*100:.2f}%)")
    print(f"  Median relative error: {median_err:.4f} ({median_err*100:.2f}%)")
    print(f"  DRE@0.05:              {(df['rel_error'] < 0.05).mean()*100:.1f}%")

    # Save CSV
    output_csv = args.output_csv or os.path.join(args.eval_dir, "diameter_error.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nPer-sample results saved to: {output_csv}")

    # Save summary statistics
    stats_path = output_csv.replace('.csv', '_statistics.csv')
    stats_rows = []
    for obj_id in sorted(errors_by_obj.keys()):
        obj_rows = df[df['object_id'] == obj_id]
        stats_rows.append({
            'object_id': obj_id,
            'n_samples': len(obj_rows),
            'gt_diameter_mm': obj_rows['gt_diameter_mm'].iloc[0],
            'mean_pred_diameter_mm': obj_rows['pred_diameter_mm'].mean(),
            'mean_diameter_score': obj_rows['diameter_score'].mean(),
            'median_diameter_score': obj_rows['diameter_score'].median(),
            'mean_rel_error': obj_rows['rel_error'].mean(),
            'mean_diameter_ratio': obj_rows['diameter_ratio'].mean(),
        })
    stats_rows.append({
        'object_id': 'ALL',
        'n_samples': len(df),
        'gt_diameter_mm': np.nan,
        'mean_pred_diameter_mm': df['pred_diameter_mm'].mean(),
        'mean_diameter_score': mean_score,
        'median_diameter_score': median_score,
        'mean_rel_error': mean_err,
        'mean_diameter_ratio': mean_ratio,
    })
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
    print(f"Statistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
