"""
Pose statistics loading and aggregation utilities.

Each WDS dataset folder stores its own pose_stats_<variant>.json file
(precomputed via scripts/compute_pose_stats.py).

At training time, the dataloader extracts dataset directories from
the shard pattern, loads per-folder stats, and aggregates them into
a single global normalization using weighted merging.
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

# Default quality filter thresholds for pose filtering.
FILTER_DEFAULTS = {'min_scale': 0.3, 'max_scale': 3.0, 'max_translation_norm': 4.0, 'max_pose_norm_threshold': 10.0}


def get_filter_defaults(pose_variant: str = "") -> Dict[str, float]:
    """Get a copy of the default filter thresholds."""
    return dict(FILTER_DEFAULTS)


def _extract_dataset_dirs(shards_pattern) -> List[Path]:
    """Extract dataset directories from a brace-expansion shard pattern.

    E.g., "./data/{ABO_wds/shard-{000..005},HSSD_wds/shard-{000..010}}.tar"
    returns [Path("./data/ABO_wds"), Path("./data/HSSD_wds")]

    For simple patterns like "./data/ABO_wds/*.tar", returns [Path("./data/ABO_wds")].
    """
    if '{' not in shards_pattern:
        # Simple glob like "./data/ABO_wds/*.tar" or "./data/ABO_wds/shard-*.tar"
        return [Path(shards_pattern).parent]

    # Extract base directory (everything before the first '{')
    brace_start = shards_pattern.index('{')
    base_dir = shards_pattern[:brace_start]

    # Find the matching outermost closing brace
    depth = 0
    brace_end = -1
    for i in range(brace_start, len(shards_pattern)):
        if shards_pattern[i] == '{':
            depth += 1
        elif shards_pattern[i] == '}':
            depth -= 1
            if depth == 0:
                brace_end = i
                break

    if brace_end == -1:
        return [Path(shards_pattern).parent]

    # Inside the outermost braces, split on top-level commas
    inner = shards_pattern[brace_start + 1:brace_end]
    parts = []
    depth = 0
    start = 0
    for i, c in enumerate(inner):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        elif c == ',' and depth == 0:
            parts.append(inner[start:i])
            start = i + 1
    parts.append(inner[start:])

    dirs = []
    seen = set()
    for part in parts:
        if '/' in part:
            dir_part = part.split('/')[0]
            d = Path(base_dir) / dir_part
        else:
            # Single-level brace (numeric range like {000000..013481}),
            # base_dir may include a filename prefix (e.g. ".../shard-"),
            # so use the parent directory
            d = Path(base_dir)
            if not d.is_dir():
                d = d.parent
        if d not in seen:
            seen.add(d)
            dirs.append(d)

    return dirs


def extract_dataset_names_from_shards(shards_pattern: str) -> List[str]:
    """Extract dataset directory names from any shard pattern (local or pipe URL).

    Handles both local patterns and S3 pipe URL patterns:
        "./data/{ABO_wds/shard-{...},HSSD_wds/shard-{...}}.tar"  → ["ABO_wds", "HSSD_wds"]
        "pipe:aws s3 cp s3://bucket/path/{ABO_wds/shard-{...}}.tar -"  → ["ABO_wds"]

    For simple single-dataset patterns, returns the parent directory name.
    """
    # Strip pipe: prefix and trailing arguments (e.g., " -")
    pattern = shards_pattern
    if pattern.startswith('pipe:'):
        # Extract the URL/path from the pipe command
        # Common format: "pipe:aws s3 cp s3://bucket/path/{...}.tar -"
        # Find the s3:// or http:// URL, or just use the last argument before " -"
        s3_match = re.search(r's3://\S+', pattern)
        if s3_match:
            pattern = s3_match.group(0)
            # Remove trailing " -" if present (from aws s3 cp ... -)
            pattern = pattern.rstrip()
            if pattern.endswith(' -'):
                pattern = pattern[:-2]
        else:
            # Fallback: strip "pipe:" prefix and trailing " -"
            pattern = pattern[5:].strip()
            if pattern.endswith(' -'):
                pattern = pattern[:-2].strip()

    if '{' not in pattern:
        return [Path(pattern).parent.name]

    brace_start = pattern.index('{')
    brace_end = -1
    depth = 0
    for i in range(brace_start, len(pattern)):
        if pattern[i] == '{':
            depth += 1
        elif pattern[i] == '}':
            depth -= 1
            if depth == 0:
                brace_end = i
                break

    if brace_end == -1:
        return [Path(pattern).parent.name]

    inner = pattern[brace_start + 1:brace_end]
    parts = []
    depth = 0
    start = 0
    for i, c in enumerate(inner):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        elif c == ',' and depth == 0:
            parts.append(inner[start:i])
            start = i + 1
    parts.append(inner[start:])

    names = []
    seen = set()
    for part in parts:
        if '/' in part:
            name = part.split('/')[0]
        else:
            # Single dataset with numeric range (e.g., "path/ABO_wds/shard-{000..010}.tar")
            # base_dir may include a filename prefix, so find the actual directory
            base = pattern[:brace_start]
            candidate = Path(base)
            # If base ends with a filename prefix like "shard-", go to parent
            if '/' in base or candidate.suffix:
                candidate = candidate.parent
            name = candidate.name or candidate.parent.name
        if name not in seen:
            seen.add(name)
            names.append(name)

    return names


def _merge_component_stats(all_stats: List[Dict], component: str) -> Dict:
    """Merge a single component (e.g. '6d_rotation') across multiple stats dicts.

    Uses the parallel variance formula:
        combined_mean = sum(n_i * mean_i) / N
        combined_var  = sum(n_i * (var_i + mean_i^2)) / N - combined_mean^2
    """
    total_count = 0
    weighted_mean_sum = None
    weighted_var_plus_mean2_sum = None

    for stats in all_stats:
        gs = stats['global_statistics'][component]
        n = gs['count']
        mean = np.array(gs['mean'], dtype=np.float64)
        std = np.array(gs['std'], dtype=np.float64)
        var = std ** 2

        if weighted_mean_sum is None:
            weighted_mean_sum = n * mean
            weighted_var_plus_mean2_sum = n * (var + mean ** 2)
        else:
            weighted_mean_sum += n * mean
            weighted_var_plus_mean2_sum += n * (var + mean ** 2)
        total_count += n

    combined_mean = weighted_mean_sum / total_count
    combined_var = weighted_var_plus_mean2_sum / total_count - combined_mean ** 2
    combined_std = np.sqrt(np.maximum(combined_var, 0.0))
    # Clamp very small stds to 1.0 to avoid division by zero
    combined_std = np.where(combined_std < 1e-8, 1.0, combined_std)

    return {
        'mean': combined_mean.tolist(),
        'std': combined_std.tolist(),
        'count': total_count,
    }


def _merge_scalar_stats(all_stats: List[Dict], component: str) -> Dict:
    """Merge a scalar component (e.g. 'scale') across multiple stats dicts."""
    total_count = 0
    weighted_mean_sum = 0.0
    weighted_var_plus_mean2_sum = 0.0

    for stats in all_stats:
        gs = stats['global_statistics'][component]
        n = gs['count']
        mean = float(gs['mean'])
        std = float(gs['std'])
        var = std ** 2

        weighted_mean_sum += n * mean
        weighted_var_plus_mean2_sum += n * (var + mean ** 2)
        total_count += n

    combined_mean = weighted_mean_sum / total_count
    combined_var = weighted_var_plus_mean2_sum / total_count - combined_mean ** 2
    combined_std = max(float(np.sqrt(max(combined_var, 0.0))), 1e-8)

    return {
        'mean': combined_mean,
        'std': combined_std,
        'count': total_count,
    }


def aggregate_stats(all_stats: List[Dict]) -> Dict:
    """Aggregate multiple per-dataset stats into a single global stats dict.

    Args:
        all_stats: List of stats dicts (each with 'global_statistics')

    Returns:
        Merged stats dict with the same structure as individual stats.
    """
    merged_global = {}
    for component in ['6d_rotation', 'quaternion', '9d_rotation', 'translation']:
        merged_global[component] = _merge_component_stats(all_stats, component)
    merged_global['scale'] = _merge_scalar_stats(all_stats, 'scale')

    total_samples = sum(s.get('total_samples', 0) for s in all_stats)
    total_filtered = sum(s.get('total_filtered', 0) for s in all_stats)
    total_skipped = sum(s.get('total_skipped', 0) for s in all_stats)

    return {
        'version': '1.0',
        'total_samples': total_samples,
        'total_filtered': total_filtered,
        'total_skipped': total_skipped,
        'global_statistics': merged_global,
        'aggregated_from': [s.get('source_dir', 'unknown') for s in all_stats],
    }


def get_or_compute_pose_stats(
    shards_pattern: str,
    pose_variant: str = "minmax",
    num_workers: int = 8,
    min_scale: Optional[float] = None,
    max_scale: Optional[float] = None,
    max_translation_norm: Optional[float] = None,
    verbose: bool = True,
    dataset_dirs: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Load per-folder pose stats and aggregate them.

    Looks for pose_stats_<variant>.json in each dataset directory extracted
    from the shard pattern (root or additional_metadata/). If all files exist,
    aggregates them. If any are missing, raises FileNotFoundError with
    instructions — precomputed stats are required for reproducible training.

    Args:
        shards_pattern: The shard pattern string
        pose_variant: The pose variant
        num_workers: Unused (kept for signature compatibility)
        min_scale: Minimum valid scale for filtering (auto-detected from variant if None)
        max_scale: Maximum valid scale for filtering (default 3.0)
        max_translation_norm: Maximum translation norm for filtering (default 4.0)
        verbose: Whether to print status messages
        dataset_dirs: Pre-resolved dataset directories (overrides shard pattern parsing)

    Returns:
        Aggregated statistics dictionary
    """
    # Auto-detect filter defaults based on depth type
    defaults = get_filter_defaults(pose_variant)
    if min_scale is None:
        min_scale = defaults['min_scale']
    if max_scale is None:
        max_scale = defaults['max_scale']
    if max_translation_norm is None:
        max_translation_norm = defaults['max_translation_norm']
    if dataset_dirs is None:
        dataset_dirs = _extract_dataset_dirs(shards_pattern)
    stats_filename = f"pose_stats_{pose_variant}.json"

    if verbose:
        print(f"Looking for per-folder pose stats ({stats_filename}) in {len(dataset_dirs)} directories...")

    # Try to load per-folder stats (check both root and additional_metadata/)
    all_stats = []
    missing_dirs = []
    for d in dataset_dirs:
        stats_path = d / stats_filename
        stats_path_alt = d / "additional_metadata" / stats_filename
        if stats_path.exists():
            found_path = stats_path
        elif stats_path_alt.exists():
            found_path = stats_path_alt
        else:
            found_path = None

        if found_path is not None:
            with open(found_path) as f:
                stats = json.load(f)
            stats['source_dir'] = str(d)
            all_stats.append(stats)
            if verbose:
                n = stats.get('total_samples', '?')
                print(f"  Loaded: {found_path} ({n} samples)")
        else:
            missing_dirs.append(d)
            if verbose:
                print(f"  Missing: {stats_path}")

    if missing_dirs:
        raise FileNotFoundError(
            f"Missing pose stats file(s) {stats_filename} in: "
            f"{[str(d) for d in missing_dirs]}. Either precompute them with "
            f"scripts/compute_pose_stats.py or pass 'pose_stats_file' in the "
            f"dataset config to load pre-aggregated statistics directly."
        )

    # All per-folder stats found — aggregate
    if verbose:
        print(f"Aggregating stats from {len(all_stats)} datasets...")

    merged = aggregate_stats(all_stats)
    merged['pose_variant'] = pose_variant

    if verbose:
        gs = merged['global_statistics']
        print(f"  Total samples: {merged['total_samples']}")
        print(f"  Scale: mean={gs['scale']['mean']:.4f}, std={gs['scale']['std']:.4f}")
        print(f"  Translation mean: {[f'{x:.4f}' for x in gs['translation']['mean']]}")

    return merged


def _weighted_merge_component(stats_a: Dict, stats_b: Dict, w_a: float, w_b: float, component: str) -> Dict:
    """Merge a single component from two stats dicts using mixture distribution weights.

    Uses the mixture distribution formula:
        combined_mean = w_a * mean_a + w_b * mean_b
        combined_var  = w_a * (var_a + mean_a²) + w_b * (var_b + mean_b²) - combined_mean²
    """
    gs_a = stats_a['global_statistics'][component]
    gs_b = stats_b['global_statistics'][component]

    mean_a = np.array(gs_a['mean'], dtype=np.float64)
    mean_b = np.array(gs_b['mean'], dtype=np.float64)
    std_a = np.array(gs_a['std'], dtype=np.float64)
    std_b = np.array(gs_b['std'], dtype=np.float64)
    var_a, var_b = std_a ** 2, std_b ** 2

    combined_mean = w_a * mean_a + w_b * mean_b
    combined_var = w_a * (var_a + mean_a ** 2) + w_b * (var_b + mean_b ** 2) - combined_mean ** 2
    combined_std = np.sqrt(np.maximum(combined_var, 0.0))
    combined_std = np.where(combined_std < 1e-8, 1.0, combined_std)

    combined_count = int(w_a * gs_a['count'] + w_b * gs_b['count'])
    return {'mean': combined_mean.tolist(), 'std': combined_std.tolist(), 'count': combined_count}


def _weighted_merge_scalar(stats_a: Dict, stats_b: Dict, w_a: float, w_b: float, component: str) -> Dict:
    """Merge a scalar component from two stats dicts using mixture distribution weights."""
    gs_a = stats_a['global_statistics'][component]
    gs_b = stats_b['global_statistics'][component]

    mean_a, mean_b = float(gs_a['mean']), float(gs_b['mean'])
    var_a, var_b = float(gs_a['std']) ** 2, float(gs_b['std']) ** 2

    combined_mean = w_a * mean_a + w_b * mean_b
    combined_var = w_a * (var_a + mean_a ** 2) + w_b * (var_b + mean_b ** 2) - combined_mean ** 2
    combined_std = max(float(np.sqrt(max(combined_var, 0.0))), 1e-8)

    combined_count = int(w_a * gs_a['count'] + w_b * gs_b['count'])
    return {'mean': combined_mean, 'std': combined_std, 'count': combined_count}


def _load_stereo_stats_available(
    dataset_dirs: List[Path],
    st_variant: str,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """Load stereo stats only from dataset dirs that have them.

    Unlike get_or_compute_pose_stats, this does NOT raise for datasets without
    stereo depth. Dirs without stereo stats files are simply skipped.

    Returns aggregated stereo stats, or None if no dirs have stereo stats.
    """
    stats_filename = f"pose_stats_{st_variant}.json"
    all_stats = []

    for d in dataset_dirs:
        stats_path = d / stats_filename
        stats_path_alt = d / "additional_metadata" / stats_filename
        if stats_path.exists():
            found_path = stats_path
        elif stats_path_alt.exists():
            found_path = stats_path_alt
        else:
            if verbose:
                print(f"  No stereo stats in {d.name} (skipped)")
            continue

        with open(found_path) as f:
            stats = json.load(f)
        stats['source_dir'] = str(d)
        all_stats.append(stats)
        if verbose:
            n = stats.get('total_samples', '?')
            print(f"  Loaded stereo: {found_path} ({n} samples)")

    if not all_stats:
        return None

    return aggregate_stats(all_stats)


def load_weighted_pose_stats(
    shards_pattern: str,
    pose_variant: str,
    num_workers: int = 8,
    verbose: bool = True,
    dataset_dirs: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Load stats for both normal and stereo depth variants and merge with data-driven weights.

    When mixing stereo/normal depth with p=0.5 for samples that have both:
        f = N_stereo_views / N_total_views  (fraction with stereo available)
        w_normal = 1 - 0.5*f
        w_st = 0.5*f

    f is approximated as stereo_stats.count / normal_stats.count since normal
    stats cover all views and stereo stats cover only views with stereo depth.

    Args:
        shards_pattern: The shard pattern string
        pose_variant: The BASE pose variant (e.g. "median_quantile_5per", without "st_" prefix)
        num_workers: Unused (kept for signature compatibility)
        verbose: Whether to print status messages
        dataset_dirs: Pre-resolved dataset directories (overrides shard pattern parsing)

    Returns:
        Weighted-merged statistics dictionary
    """
    base_variant = pose_variant[3:] if pose_variant.startswith('st_') else pose_variant
    st_variant = f"st_{base_variant}"

    if verbose:
        print(f"Loading weighted pose stats: normal={base_variant}, st={st_variant}")

    normal_stats = get_or_compute_pose_stats(
        shards_pattern, base_variant, num_workers=num_workers, verbose=verbose,
        dataset_dirs=dataset_dirs,
    )

    # Load stereo stats only from dirs that have them (no expensive fallback)
    if dataset_dirs is None:
        dataset_dirs = _extract_dataset_dirs(shards_pattern)
    st_stats = _load_stereo_stats_available(dataset_dirs, st_variant, verbose=verbose)

    if st_stats is None:
        if verbose:
            print("  No stereo stats found in any dataset — using normal stats only.")
        normal_stats['pose_variant'] = base_variant
        return normal_stats

    # Compute f = fraction of views with stereo available
    n_normal = normal_stats['global_statistics']['scale']['count']
    n_stereo = st_stats['global_statistics']['scale']['count']

    if n_normal == 0:
        raise ValueError("Normal pose stats have zero count; cannot compute mixture weights.")

    f = n_stereo / n_normal
    w_normal = 1.0 - 0.5 * f
    w_st = 0.5 * f

    if verbose:
        print(f"  Stereo fraction f = {n_stereo}/{n_normal} = {f:.4f}")
        print(f"  Mixture weights: w_normal={w_normal:.4f}, w_st={w_st:.4f}")

    # Merge each component
    merged_global = {}
    for component in ['6d_rotation', 'quaternion', '9d_rotation', 'translation']:
        merged_global[component] = _weighted_merge_component(normal_stats, st_stats, w_normal, w_st, component)
    merged_global['scale'] = _weighted_merge_scalar(normal_stats, st_stats, w_normal, w_st, 'scale')

    merged = {
        'version': '1.0',
        'total_samples': normal_stats.get('total_samples', 0) + st_stats.get('total_samples', 0),
        'global_statistics': merged_global,
        'pose_variant': f"{base_variant}+st_{base_variant}",
        'stereo_fraction': f,
        'w_normal': w_normal,
        'w_st': w_st,
        'normal_stats_summary': {
            'total_samples': normal_stats.get('total_samples', 0),
            'count': n_normal,
            'scale_mean': normal_stats['global_statistics']['scale']['mean'],
        },
        'st_stats_summary': {
            'total_samples': st_stats.get('total_samples', 0),
            'count': n_stereo,
            'scale_mean': st_stats['global_statistics']['scale']['mean'],
        },
    }

    if verbose:
        gs = merged_global
        print(f"  Combined scale: mean={gs['scale']['mean']:.4f}, std={gs['scale']['std']:.4f}")
        print(f"  Combined trans mean: {[f'{x:.4f}' for x in gs['translation']['mean']]}")

    return merged
