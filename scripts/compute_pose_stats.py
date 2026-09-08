"""
Compute pose normalization statistics from WebDataset shards.

This script streams through all WebDataset shards and computes:
- 6D rotation statistics (mean, std for 6 components)
- Translation statistics (mean, std for 3 components)
- Scale statistics (mean, std)

Both global statistics (across all datasets) and per-dataset statistics are computed.

Supports parallel processing with multiple workers for faster computation.
Each worker extracts pose data and saves to temporary .npy files,
then a final pass computes statistics from all saved arrays.

Usage:
    python scripts/compute_pose_normalization_stats.py \
        --output configs/pose_normalization_stats.json \
        --compute_per_dataset \
        --num_workers 8

    # With custom shards (e.g., single dataset):
    python scripts/compute_pose_normalization_stats.py \
        --shards "./datasets/wds_multiview_robust_pose/ABO_wds/*.tar" \
        --output configs/pose_normalization_stats_abo.json

    # Multiple datasets:
    python scripts/compute_pose_normalization_stats.py \
        --shards "./datasets/wds_multiview_robust_pose/ABO_wds/*.tar" \
                 "./datasets/wds_multiview_robust_pose/HSSD_wds/*.tar" \
        --output configs/pose_normalization_stats_test.json
"""

import argparse
import gc
import io
import json
import os
import sys
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
import numpy as np
import webdataset as wds
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from recgen_inference.recgen_modules.utils.pose_utils import save_pose_normalization_stats


def extract_dataset_name(shard_path: str) -> str:
    """Extract dataset name from shard path."""
    parts = Path(shard_path).parts
    for part in parts:
        if 'ABO' in part:
            return 'ABO'
        elif 'HSSD' in part:
            return 'HSSD'
        elif 'Objaverse' in part:
            return 'Objaverse'
        elif 'PhysX3DParts' in part:
            return 'PhysX3DParts'
        elif 'PartNeXt' in part:
            return 'PartNeXt'
    return 'unknown'


def get_shard_list(shard_patterns: List[str]) -> List[str]:
    """Get list of shard files from glob patterns.

    Args:
        shard_patterns: List of glob patterns (e.g., ["./data/ABO_wds/*.tar", "./data/HSSD_wds/*.tar"])
    """
    shard_files = []
    for pattern in shard_patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"WARNING: no files matched pattern: {pattern}")
        shard_files.extend(matched)

    return shard_files


def _empty_running_stats():
    """Create empty running statistics accumulators (sum, sum_sq, count) for each pose component."""
    return {
        'rot_6d_sum': np.zeros(6, dtype=np.float64),
        'rot_6d_sum_sq': np.zeros(6, dtype=np.float64),
        'quat_sum': np.zeros(4, dtype=np.float64),
        'quat_sum_sq': np.zeros(4, dtype=np.float64),
        'rot_9d_sum': np.zeros(9, dtype=np.float64),
        'rot_9d_sum_sq': np.zeros(9, dtype=np.float64),
        'trans_sum': np.zeros(3, dtype=np.float64),
        'trans_sum_sq': np.zeros(3, dtype=np.float64),
        'scale_sum': 0.0,
        'scale_sum_sq': 0.0,
        'count': 0,
    }


def _accumulate(stats, rot_6d, quat, rot_9d, trans, scale):
    """Add a single pose sample to running statistics."""
    stats['rot_6d_sum'] += rot_6d
    stats['rot_6d_sum_sq'] += rot_6d ** 2
    stats['quat_sum'] += quat
    stats['quat_sum_sq'] += quat ** 2
    stats['rot_9d_sum'] += rot_9d
    stats['rot_9d_sum_sq'] += rot_9d ** 2
    stats['trans_sum'] += trans
    stats['trans_sum_sq'] += trans ** 2
    stats['scale_sum'] += scale
    stats['scale_sum_sq'] += scale ** 2
    stats['count'] += 1


def _merge_running_stats(a, b):
    """Merge two running statistics dicts."""
    return {
        'rot_6d_sum': a['rot_6d_sum'] + b['rot_6d_sum'],
        'rot_6d_sum_sq': a['rot_6d_sum_sq'] + b['rot_6d_sum_sq'],
        'quat_sum': a['quat_sum'] + b['quat_sum'],
        'quat_sum_sq': a['quat_sum_sq'] + b['quat_sum_sq'],
        'rot_9d_sum': a['rot_9d_sum'] + b['rot_9d_sum'],
        'rot_9d_sum_sq': a['rot_9d_sum_sq'] + b['rot_9d_sum_sq'],
        'trans_sum': a['trans_sum'] + b['trans_sum'],
        'trans_sum_sq': a['trans_sum_sq'] + b['trans_sum_sq'],
        'scale_sum': a['scale_sum'] + b['scale_sum'],
        'scale_sum_sq': a['scale_sum_sq'] + b['scale_sum_sq'],
        'count': a['count'] + b['count'],
    }


def _finalize_stats(stats):
    """Compute mean and std from running statistics."""
    n = stats['count']
    if n == 0:
        return None

    def _mean_std(s, s2):
        mean = s / n
        var = s2 / n - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0))
        std = np.where(std < 1e-8, 1.0, std)
        return mean, std

    rot_6d_mean, rot_6d_std = _mean_std(stats['rot_6d_sum'], stats['rot_6d_sum_sq'])
    quat_mean, quat_std = _mean_std(stats['quat_sum'], stats['quat_sum_sq'])
    rot_9d_mean, rot_9d_std = _mean_std(stats['rot_9d_sum'], stats['rot_9d_sum_sq'])
    trans_mean, trans_std = _mean_std(stats['trans_sum'], stats['trans_sum_sq'])
    scale_mean = stats['scale_sum'] / n
    scale_var = stats['scale_sum_sq'] / n - scale_mean ** 2
    scale_std = max(float(np.sqrt(max(scale_var, 0.0))), 1e-8)

    return {
        '6d_rotation': {'mean': rot_6d_mean.tolist(), 'std': rot_6d_std.tolist(), 'count': n},
        'quaternion': {'mean': quat_mean.tolist(), 'std': quat_std.tolist(), 'count': n},
        '9d_rotation': {'mean': rot_9d_mean.tolist(), 'std': rot_9d_std.tolist(), 'count': n},
        'translation': {'mean': trans_mean.tolist(), 'std': trans_std.tolist(), 'count': n},
        'scale': {'mean': float(scale_mean), 'std': float(scale_std), 'count': n},
    }


def process_single_shard(args: Tuple) -> Dict:
    """Process a single shard file and return running statistics.

    Uses online accumulation (sum, sum_sq, count) so memory is O(1) per shard.
    """
    (shard_path, temp_dir, shard_idx, compute_per_dataset, min_visible_fraction,
     min_visible_pixels, max_pose_norm_threshold, min_scale, max_scale,
     max_translation_norm, pose_variant) = args

    # Determine pose data key based on variant
    if pose_variant == "minmax":
        pose_key = "pose_data.json"
    else:
        pose_key = f"pose_data_{pose_variant}.json"

    # Running statistics accumulators (constant memory)
    global_stats = _empty_running_stats()
    per_dataset_stats = {}  # dataset_name -> running stats

    num_samples = 0
    num_skipped = 0
    num_filtered = 0

    try:
        dataset = wds.WebDataset(
            shard_path,
            shardshuffle=False,
            nodesplitter=wds.shardlists.split_by_worker,
            handler=wds.handlers.warn_and_continue
        )

        for sample in dataset:
            try:
                if pose_key not in sample:
                    num_skipped += 1
                    continue

                # Extract dataset name
                dataset_name = 'unknown'
                if '__url__' in sample:
                    dataset_name = extract_dataset_name(sample['__url__'])

                # Load view_metadata for visible_fraction filtering
                view_metadata_list = None
                if 'view_metadata.json' in sample:
                    meta_raw = sample['view_metadata.json']
                    if isinstance(meta_raw, bytes):
                        meta_raw = meta_raw.decode('utf-8')
                    view_metadata_list = json.loads(meta_raw)
                    if not isinstance(view_metadata_list, list):
                        view_metadata_list = [view_metadata_list]

                # Collect raw mask bytes — decoded lazily during filtering
                # to avoid holding ~5.5 MB of numpy arrays per sample.
                mask_raw = {}
                for key in sample.keys():
                    if key.startswith('cond_mask_') and key.endswith('.png') and '_full_' not in key:
                        try:
                            idx_str = key.replace('cond_mask_', '').replace('.png', '')
                            mask_raw[int(idx_str)] = sample[key]
                        except ValueError:
                            pass

                pose_data_raw = sample[pose_key]
                if isinstance(pose_data_raw, bytes):
                    pose_data_str = pose_data_raw.decode('utf-8')
                else:
                    pose_data_str = pose_data_raw

                pose_data = json.loads(pose_data_str)
                if isinstance(pose_data, dict):
                    pose_data_list = [pose_data]
                elif isinstance(pose_data, list):
                    pose_data_list = pose_data
                else:
                    num_skipped += 1
                    continue

                # Process each view
                for view_idx, pose_dict in enumerate(pose_data_list):
                    scale = pose_dict['scale']
                    trans = np.array([
                        pose_dict['trans_x'],
                        pose_dict['trans_y'],
                        pose_dict['trans_z']
                    ], dtype=np.float64)

                    # === Quality filtering ===

                    # 1. Check visible_fraction
                    if view_metadata_list is not None and view_idx < len(view_metadata_list):
                        try:
                            meta = view_metadata_list[view_idx]
                            if isinstance(meta, str):
                                meta = json.loads(meta)
                            visible_fraction = meta.get('visible_fraction', 1.0)
                            if visible_fraction < min_visible_fraction:
                                num_filtered += 1
                                continue
                        except (json.JSONDecodeError, TypeError, KeyError):
                            pass

                    # 2. Check mask pixel count (decode lazily)
                    if view_idx in mask_raw:
                        mask_img = Image.open(io.BytesIO(mask_raw[view_idx]))
                        mask_pixel_count = np.count_nonzero(np.array(mask_img))
                        mask_img.close()
                        if mask_pixel_count <= min_visible_pixels:
                            num_filtered += 1
                            continue

                    # 3. Check scale bounds
                    if scale <= min_scale or scale >= max_scale:
                        num_filtered += 1
                        continue

                    # 4. Check translation norm
                    translation_norm = np.linalg.norm(trans)
                    if translation_norm > max_translation_norm:
                        num_filtered += 1
                        continue

                    # 5. Check pose norm
                    rot_6d = np.array([
                        pose_dict['rot6d_0'], pose_dict['rot6d_1'], pose_dict['rot6d_2'],
                        pose_dict['rot6d_3'], pose_dict['rot6d_4'], pose_dict['rot6d_5']
                    ], dtype=np.float64)
                    pose_10d = np.concatenate([rot_6d, trans, [scale]])
                    pose_norm = np.linalg.norm(pose_10d)
                    if pose_norm > max_pose_norm_threshold:
                        num_filtered += 1
                        continue

                    # === Passed all filters - accumulate running stats ===
                    quat = np.array([
                        pose_dict['quat_x'], pose_dict['quat_y'],
                        pose_dict['quat_z'], pose_dict['quat_w']
                    ], dtype=np.float64)

                    rot_9d = np.array([
                        pose_dict['rot9d_0'], pose_dict['rot9d_1'], pose_dict['rot9d_2'],
                        pose_dict['rot9d_3'], pose_dict['rot9d_4'], pose_dict['rot9d_5'],
                        pose_dict['rot9d_6'], pose_dict['rot9d_7'], pose_dict['rot9d_8']
                    ], dtype=np.float64)

                    _accumulate(global_stats, rot_6d, quat, rot_9d, trans, float(scale))

                    if compute_per_dataset:
                        if dataset_name not in per_dataset_stats:
                            per_dataset_stats[dataset_name] = _empty_running_stats()
                        _accumulate(per_dataset_stats[dataset_name], rot_6d, quat, rot_9d, trans, float(scale))

                    num_samples += 1

            except Exception as e:
                num_skipped += 1
                continue
            finally:
                # Release per-sample objects to prevent memory fragmentation
                del mask_raw

        # Force garbage collection after each shard to release fragmented memory
        gc.collect()

    except Exception as e:
        print(f"Error processing shard {shard_path}: {e}")
        return None

    return {
        'global_stats': global_stats,
        'per_dataset_stats': per_dataset_stats,
        'num_samples': num_samples,
        'num_skipped': num_skipped,
        'num_filtered': num_filtered,
    }


def compute_final_statistics(shard_results: List[Dict], compute_per_dataset: bool) -> Dict:
    """Merge running statistics from all shard workers into final output.

    This is O(num_shards) in time and O(num_datasets) in memory — no large arrays.
    """
    print("Merging running statistics from all shards...")

    merged_global = _empty_running_stats()
    merged_per_dataset = {}  # dataset_name -> running stats

    total_samples = 0
    total_skipped = 0
    total_filtered = 0

    for result in shard_results:
        if result is None:
            continue
        total_samples += result['num_samples']
        total_skipped += result['num_skipped']
        total_filtered += result['num_filtered']

        if result['num_samples'] > 0:
            merged_global = _merge_running_stats(merged_global, result['global_stats'])

            if compute_per_dataset:
                for ds_name, ds_stats in result['per_dataset_stats'].items():
                    if ds_name not in merged_per_dataset:
                        merged_per_dataset[ds_name] = _empty_running_stats()
                    merged_per_dataset[ds_name] = _merge_running_stats(merged_per_dataset[ds_name], ds_stats)

    total_valid = total_samples + total_filtered
    print(f"Total processed: {total_samples} samples (skipped {total_skipped}, filtered {total_filtered})")
    if total_valid > 0:
        print(f"Filtering removed {total_filtered / total_valid * 100:.1f}% of pose samples")

    if total_samples == 0:
        raise ValueError("No valid samples found!")

    # Finalize global statistics
    print("Computing final statistics from running accumulators...")
    global_final = _finalize_stats(merged_global)

    output = {
        'version': '1.0',
        'computed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': total_samples,
        'total_skipped': total_skipped,
        'total_filtered': total_filtered,
        'global_statistics': global_final,
    }

    # Finalize per-dataset statistics
    if compute_per_dataset:
        print("Finalizing per-dataset statistics...")
        output['per_dataset_stats'] = {}
        for ds_name in sorted(merged_per_dataset.keys()):
            ds_final = _finalize_stats(merged_per_dataset[ds_name])
            if ds_final is not None:
                output['per_dataset_stats'][ds_name] = {
                    '6d_rotation': {'mean': ds_final['6d_rotation']['mean'], 'std': ds_final['6d_rotation']['std']},
                    'quaternion': {'mean': ds_final['quaternion']['mean'], 'std': ds_final['quaternion']['std']},
                    '9d_rotation': {'mean': ds_final['9d_rotation']['mean'], 'std': ds_final['9d_rotation']['std']},
                    'translation': {'mean': ds_final['translation']['mean'], 'std': ds_final['translation']['std']},
                    'scale': {'mean': ds_final['scale']['mean'], 'std': ds_final['scale']['std']},
                    'num_samples': ds_final['6d_rotation']['count'],
                }
        output['datasets'] = sorted(output['per_dataset_stats'].keys())
    else:
        output['datasets'] = []

    return output


def compute_stats_parallel(
    shard_patterns: List[str],
    num_workers: int = 1,
    compute_per_dataset: bool = False,
    min_visible_fraction: float = 0.2,
    min_visible_pixels: int = 400,
    max_pose_norm_threshold: float = 10.0,
    min_scale: float = 0.3,
    max_scale: float = 3.0,
    max_translation_norm: float = 4.0,
    keep_temp_dir: bool = False,
    temp_dir_path: Optional[str] = None,
    pose_variant: str = "minmax"
) -> Dict:
    """
    Compute pose statistics from WebDataset shards using parallel processing.

    Uses online statistics (sum, sum_sq, count) so memory is O(1) per worker,
    regardless of dataset size. No temp files are written.

    Args:
        shard_patterns: List of glob patterns for shard files.
        keep_temp_dir: Deprecated, kept for API compatibility.
        temp_dir_path: Deprecated, kept for API compatibility.
    """
    print(f"Loading shards from: {shard_patterns}")
    print(f"Pose variant: {pose_variant}")
    print(f"Quality filters: min_visible_fraction={min_visible_fraction}, min_visible_pixels={min_visible_pixels}")
    print(f"                 max_pose_norm={max_pose_norm_threshold}, scale=[{min_scale}, {max_scale}], max_trans_norm={max_translation_norm}")

    # Get list of shard files
    shard_files = get_shard_list(shard_patterns)
    print(f"Found {len(shard_files)} shard files")

    if len(shard_files) == 0:
        raise ValueError(f"No shard files found matching patterns: {shard_patterns}")

    # temp_dir is unused but kept in the arg tuple for process_single_shard signature compatibility
    temp_dir = ""

    # Prepare arguments for each shard
    args_list = [
        (shard_path, temp_dir, idx, compute_per_dataset, min_visible_fraction,
         min_visible_pixels, max_pose_norm_threshold, min_scale, max_scale,
         max_translation_norm, pose_variant)
        for idx, shard_path in enumerate(shard_files)
    ]

    # Process shards and merge results incrementally to avoid holding all
    # results in memory.  Each worker returns a small running-stats dict;
    # we merge it immediately and discard it.
    print(f"\nProcessing {len(shard_files)} shards with {num_workers} workers (streaming stats)...")

    merged_global = _empty_running_stats()
    merged_per_dataset = {}
    total_samples = 0
    total_skipped = 0
    total_filtered = 0

    def _merge_result(result):
        nonlocal merged_global, merged_per_dataset, total_samples, total_skipped, total_filtered
        if result is None:
            return
        total_samples += result['num_samples']
        total_skipped += result['num_skipped']
        total_filtered += result['num_filtered']
        if result['num_samples'] > 0:
            merged_global = _merge_running_stats(merged_global, result['global_stats'])
            if compute_per_dataset:
                for ds_name, ds_stats in result['per_dataset_stats'].items():
                    if ds_name not in merged_per_dataset:
                        merged_per_dataset[ds_name] = _empty_running_stats()
                    merged_per_dataset[ds_name] = _merge_running_stats(
                        merged_per_dataset[ds_name], ds_stats)

    if num_workers == 1:
        for args in tqdm(args_list, desc="Processing shards"):
            _merge_result(process_single_shard(args))
    else:
        # maxtasksperchild restarts workers periodically to free leaked memory
        with Pool(num_workers, maxtasksperchild=20) as pool:
            for result in tqdm(
                pool.imap_unordered(process_single_shard, args_list),
                total=len(args_list),
                desc="Processing shards"
            ):
                _merge_result(result)

    # Build the final output from the incrementally-merged stats
    print("\nFinalizing statistics...")
    total_valid = total_samples + total_filtered
    print(f"Total processed: {total_samples} samples (skipped {total_skipped}, filtered {total_filtered})")
    if total_valid > 0:
        print(f"Filtering removed {total_filtered / total_valid * 100:.1f}% of pose samples")

    if total_samples == 0:
        raise ValueError("No valid samples found!")

    global_final = _finalize_stats(merged_global)

    output = {
        'version': '1.0',
        'computed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': total_samples,
        'total_skipped': total_skipped,
        'total_filtered': total_filtered,
        'global_statistics': global_final,
    }

    if compute_per_dataset:
        output['per_dataset_stats'] = {}
        for ds_name in sorted(merged_per_dataset.keys()):
            ds_final = _finalize_stats(merged_per_dataset[ds_name])
            if ds_final is not None:
                output['per_dataset_stats'][ds_name] = {
                    '6d_rotation': {'mean': ds_final['6d_rotation']['mean'], 'std': ds_final['6d_rotation']['std']},
                    'quaternion': {'mean': ds_final['quaternion']['mean'], 'std': ds_final['quaternion']['std']},
                    '9d_rotation': {'mean': ds_final['9d_rotation']['mean'], 'std': ds_final['9d_rotation']['std']},
                    'translation': {'mean': ds_final['translation']['mean'], 'std': ds_final['translation']['std']},
                    'scale': {'mean': ds_final['scale']['mean'], 'std': ds_final['scale']['std']},
                    'num_samples': ds_final['6d_rotation']['count'],
                }
        output['datasets'] = sorted(output['per_dataset_stats'].keys())
    else:
        output['datasets'] = []

    # Add pose_variant to output
    output['pose_variant'] = pose_variant

    return output


def print_statistics_summary(stats: Dict):
    """Print a summary of computed statistics."""
    print("\n" + "=" * 80)
    print("GLOBAL STATISTICS SUMMARY")
    print("=" * 80)

    global_stats = stats['global_statistics']

    if 'pose_variant' in stats:
        print(f"\nPose variant: {stats['pose_variant']}")
    print(f"Total samples: {stats['total_samples']}")
    if 'total_filtered' in stats:
        print(f"Total filtered: {stats['total_filtered']}")
    if 'total_skipped' in stats:
        print(f"Total skipped: {stats['total_skipped']}")

    print("\n6D Rotation Statistics:")
    print(f"  Mean: {np.array(global_stats['6d_rotation']['mean'])}")
    print(f"  Std:  {np.array(global_stats['6d_rotation']['std'])}")

    print("\nTranslation Statistics:")
    print(f"  Mean: {np.array(global_stats['translation']['mean'])}")
    print(f"  Std:  {np.array(global_stats['translation']['std'])}")

    print("\nScale Statistics:")
    print(f"  Mean: {global_stats['scale']['mean']:.4f}")
    print(f"  Std:  {global_stats['scale']['std']:.4f}")

    if 'per_dataset_stats' in stats and stats['per_dataset_stats']:
        print("\n" + "=" * 80)
        print("PER-DATASET STATISTICS")
        print("=" * 80)

        for dataset_name in sorted(stats['per_dataset_stats'].keys()):
            ds_stats = stats['per_dataset_stats'][dataset_name]
            print(f"\n{dataset_name}:")
            print(f"  Samples: {ds_stats['num_samples']}")
            print(f"  Rotation Mean: {np.array(ds_stats['6d_rotation']['mean'])}")
            print(f"  Rotation Std:  {np.array(ds_stats['6d_rotation']['std'])}")
            print(f"  Translation Mean: {np.array(ds_stats['translation']['mean'])}")
            print(f"  Translation Std:  {np.array(ds_stats['translation']['std'])}")
            print(f"  Scale Mean: {ds_stats['scale']['mean']:.4f}")
            print(f"  Scale Std:  {ds_stats['scale']['std']:.4f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Compute pose normalization statistics from WebDataset shards',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute stats from all datasets with 8 workers (uses default shards pattern)
  python scripts/compute_pose_normalization_stats.py \\
      --output configs/pose_normalization_stats.json \\
      --compute_per_dataset \\
      --num_workers 8

  # Test with specific datasets
  python scripts/compute_pose_normalization_stats.py \\
      --shards "datasets/wds_multiview_robust_pose/ABO_wds/*.tar" \\
      --output configs/pose_normalization_stats_test.json \\
      --num_workers 4
        """
    )

    parser.add_argument(
        '--shards',
        type=str,
        nargs='+',
        default=[
            './datasets/wds_multiview_robust_pose/ABO_wds/*.tar',
            './datasets/wds_multiview_robust_pose/HSSD_wds/*.tar',
            './datasets/wds_multiview_robust_pose/PhysX3DParts_wds/*.tar',
            './datasets/wds_multiview_robust_pose/Objaverse_recgen_new_wds/*.tar',
            './datasets/wds_multiview_robust_pose/PartNeXt_wds_clean/*.tar',
        ],
        help='Glob patterns for WebDataset shard files (e.g., "./data/ABO_wds/*.tar" "./data/HSSD_wds/*.tar")'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for JSON statistics file'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )
    parser.add_argument(
        '--compute_per_dataset',
        action='store_true',
        help='Compute per-dataset statistics in addition to global statistics'
    )

    # Quality filtering parameters
    parser.add_argument(
        '--min_visible_fraction',
        type=float,
        default=0.2,
        help='Minimum visible fraction of object (default: 0.2)'
    )
    parser.add_argument(
        '--min_visible_pixels',
        type=int,
        default=400,
        help='Minimum number of mask pixels (default: 400)'
    )
    parser.add_argument(
        '--max_pose_norm',
        type=float,
        default=10.0,
        help='Maximum L2 norm of full pose vector (default: 10.0)'
    )
    parser.add_argument(
        '--min_scale',
        type=float,
        default=None,
        help='Minimum valid scale (default: auto-detect from variant: 0.3 for normal, 0.05 for st_*)'
    )
    parser.add_argument(
        '--max_scale',
        type=float,
        default=3.0,
        help='Maximum valid scale (default: 3.0)'
    )
    parser.add_argument(
        '--max_translation_norm',
        type=float,
        default=4.0,
        help='Maximum L2 norm of translation (default: 4.0)'
    )
    parser.add_argument(
        '--keep_temp_dir',
        action='store_true',
        help='Keep temporary directory with extracted .npy files for debugging'
    )
    parser.add_argument(
        '--temp_dir',
        type=str,
        default=None,
        help='Custom path for temporary directory (default: system temp)'
    )
    parser.add_argument(
        '--pose_variant',
        type=str,
        default='minmax',
        choices=[
            'minmax', 'median_quantile_2.5per', 'median_quantile_5per', 'median_quantile_10per',
            'st_minmax', 'st_median_quantile_2.5per', 'st_median_quantile_5per', 'st_median_quantile_10per',
        ],
        help='Pose variant to compute stats for (default: minmax)'
    )

    args = parser.parse_args()

    # Auto-detect filter defaults from variant if not explicitly set
    from recgen_training.datasets.pose_stats import get_filter_defaults
    defaults = get_filter_defaults(args.pose_variant)
    if args.min_scale is None:
        args.min_scale = defaults['min_scale']
        print(f"Auto-detected min_scale={args.min_scale} for variant '{args.pose_variant}'")

    # Compute statistics
    stats = compute_stats_parallel(
        shard_patterns=args.shards,
        num_workers=args.num_workers,
        compute_per_dataset=args.compute_per_dataset,
        min_visible_fraction=args.min_visible_fraction,
        min_visible_pixels=args.min_visible_pixels,
        max_pose_norm_threshold=args.max_pose_norm,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        max_translation_norm=args.max_translation_norm,
        keep_temp_dir=args.keep_temp_dir,
        temp_dir_path=args.temp_dir,
        pose_variant=args.pose_variant
    )

    # Print summary
    print_statistics_summary(stats)

    # Save to file
    save_pose_normalization_stats(stats, args.output)

    print(f"\nStatistics saved to: {args.output}")
    print("Done!")


if __name__ == '__main__':
    main()
