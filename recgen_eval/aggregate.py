"""Per-sample metrics.json handling and CSV aggregation.

Ported verbatim from the internal ``evaluate_recgen_multiview.py`` (the
``aggregate_from_json`` family). The aggregated
``predictions/final_results_from_json/0_all_frames_metrics_results_statistics.csv``
MEAN row is the source of the paper-table numbers.

Also runnable standalone to (re-)aggregate an existing run directory:

    python -m recgen_eval.aggregate --run_dir <...>/dataset_HB_multiview1
"""

import argparse
import glob as glob_module
import json
import os

import numpy as np
import pandas as pd
import trimesh


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _build_mesh_for_pose_k(obj_folder, pose_k_dict, cam2ncam_k):
    """Build a mesh in camera space using pose k and cam2ncam k.

    Loads the raw GLB from obj_folder/output.glb, applies the pose k transform
    (rotation*scale + translation + 90deg GLB fix), then undoes cam2ncam_k
    normalization. Returns a trimesh object positioned in camera k's frame,
    or None if the GLB doesn't exist.
    """
    from scipy.spatial.transform import Rotation as Rot
    glb_path = os.path.join(obj_folder, 'output.glb')
    if not os.path.exists(glb_path):
        return None

    try:
        mesh = trimesh.load(glb_path, force='mesh')
    except Exception:
        scene = trimesh.load(glb_path)
        if isinstance(scene, trimesh.Scene):
            mesh = scene.to_geometry() if hasattr(scene, 'to_geometry') else scene.dump(concatenate=True)
        else:
            mesh = scene

    # trimesh 4.x cannot sample colors from PBR-textured GLBs ('PBRMaterial'
    # has no .image); bake the texture into vertex colors so the optional
    # color-chamfer inside compute_metrics_anchor works. Geometry unchanged.
    try:
        if hasattr(mesh.visual, 'to_color'):
            mesh.visual = mesh.visual.to_color()
    except Exception:
        pass

    # Build transform from parsed pose dict (same logic as get_pose in recgen_utils)
    rot_matrix = np.array(pose_k_dict['rotation_matrix'])
    t = np.array(pose_k_dict['translation'])
    scale = float(pose_k_dict['scale'])

    r = rot_matrix * scale
    r90 = Rot.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix()
    R_total = r @ r90

    T = np.eye(4)
    T[:3, :3] = R_total
    T[:3, 3] = t

    cam2ncam_k = np.array(cam2ncam_k)
    mesh.apply_transform(T)
    mesh.apply_translation(-cam2ncam_k[:3, 3])
    mesh.apply_scale(1.0 / cam2ncam_k[0, 0])

    return mesh


def _save_sample_metrics_json(metrics, obj_folder, file_name, failed=False,
                               pred_pose=None, gt_pose=None, gt_metadata=None, camera_intrinsics=None):
    """Save all metrics and poses to a JSON file at obj_folder level.

    Saves to {obj_folder}/metrics.json (always, including failures).
    NaN values are stored as null (JSON-compliant).
    """
    data = {}
    data['file_name'] = file_name
    data['_failed'] = failed

    # All scalar metrics from the metrics dict
    skip_keys = {'icp_transform', 'mssd_rec', 'mspd_rec', 'vsd_rec', 'vsd_errs', '_crop_box', '_full_icp_camera_tf'}
    for k, v in metrics.items():
        if k.startswith('_'):
            continue
        if k in skip_keys:
            continue
        # Convert NaN to None for valid JSON
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            data[k] = None
        else:
            data[k] = v

    # Poses (optional — not available for failed samples)
    if pred_pose is not None:
        data['pred_pose'] = np.array(pred_pose).tolist()
    if gt_pose is not None:
        data['gt_pose'] = np.array(gt_pose).tolist()
    if 'icp_transform' in metrics:
        data['icp_transform'] = np.array(metrics['icp_transform']).tolist()
    if '_full_icp_camera_tf' in metrics:
        data['full_icp_camera_tf'] = metrics['_full_icp_camera_tf']

    # Rendering params (optional)
    if gt_metadata is not None:
        data['gt_diameter'] = float(gt_metadata['diameter'])
    if camera_intrinsics is not None:
        data['camera_intrinsics'] = np.array(camera_intrinsics).tolist()
    if '_crop_box' in metrics:
        data['crop_box'] = metrics['_crop_box']

    os.makedirs(obj_folder, exist_ok=True)
    with open(os.path.join(obj_folder, 'metrics.json'), 'w') as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def _load_metrics_from_json(json_path):
    """Load metrics.json and convert to the dict format expected by Evaluator.add_metrics().

    Converts JSON null → np.nan for numeric fields.
    Returns (object_id, metrics_dict, file_name, failed) or None on error.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    file_name = data.get('file_name')
    failed = data.get('_failed', False)

    # Parse object_id from folder name: obj_000004_multiview_..._imgs_000040
    # For old-format (perception_renders/metrics.json), go up two levels
    parent = os.path.dirname(json_path)
    if os.path.basename(parent) == 'perception_renders':
        parent = os.path.dirname(parent)
    folder_name = os.path.basename(parent)
    try:
        parts = folder_name.split('_')
        obj_id_str = parts[1]  # "000004"
        object_id = int(obj_id_str)
        # Derive file_name from folder if not in JSON (old format)
        # Folder: obj_000004_multiview_000013_imgs_000040 → file_name = "000040.png"
        if file_name is None:
            imgs_idx = parts.index('imgs')
            first_img_id = parts[imgs_idx + 1]  # "000040"
            file_name = f"{first_img_id}.png"
    except (IndexError, ValueError):
        print(f"  [WARN] Cannot parse object_id from folder: {folder_name}")
        return None

    def _get(key, default=np.nan):
        val = data.get(key)
        if val is None:
            return default
        return val

    metrics = {
        '_failed': failed,
        'err_R': _get('err_R'), 'err_T': _get('err_T'),
        'add': _get('add'), 'adds': _get('adds'),
        'add_thres': _get('add_thres'), 'adds_thres': _get('adds_thres'),
        'mssd_err': _get('mssd_err'), 'mspd_err': _get('mspd_err'),
        'mean_mssd': _get('mean_mssd'), 'mean_mspd': _get('mean_mspd'),
        'mean_vsd': _get('mean_vsd'), 'mean_ar': _get('mean_ar'),
        # Not stored in JSON (arrays); use NaN
        'mssd_rec': np.nan, 'mspd_rec': np.nan, 'vsd_rec': np.nan, 'vsd_errs': np.nan,
        # Chamfer metrics
        'chamfer_dist': _get('chamfer_dist'), 'chamfer_dist_no_icp': _get('chamfer_dist_no_icp'),
        'chamfer_dist_icp_with_scale': _get('chamfer_dist_icp_with_scale'),
        'chamfer_dist_color': _get('chamfer_dist_color'),
        'chamfer_dist_color_no_icp': _get('chamfer_dist_color_no_icp'),
        'chamfer_normalized': _get('chamfer_normalized'),
        'chamfer_normalized_icp_with_scale': _get('chamfer_normalized_icp_with_scale'),
        'addss': _get('addss'), 'addss_10': _get('addss_10'),
        'addss_05': _get('addss_05'), 'addss_02': _get('addss_02'),
        # Perception metrics
        'lpips': _get('lpips'), 'lpips_no_icp': _get('lpips_no_icp'),
        'ssim': _get('ssim'), 'ssim_no_icp': _get('ssim_no_icp'),
        'psnr': _get('psnr'), 'psnr_no_icp': _get('psnr_no_icp'),
        'color_dist': _get('color_dist'), 'color_dist_no_icp': _get('color_dist_no_icp'),
    }

    return object_id, metrics, file_name, failed


def aggregate_from_json(save_path, config_file=None):
    """Aggregate evaluation results from per-sample metrics.json files.

    Walks save_path finding all obj_*/metrics.json files (and old-format
    perception_renders/metrics.json), builds DataFrames, and saves aggregated
    CSV results to predictions/final_results_from_json/.

    Works standalone (no Docker / bop_toolkit_lib needed).
    """
    output_dir = os.path.join(save_path, "predictions", "final_results_from_json")

    # Collect metrics.json from obj_folder level (new format)
    # and fall back to perception_renders/ (old format)
    json_files = {}  # obj_folder -> json_path (dedup by folder)
    for pattern in [
        os.path.join(save_path, "obj_*", "metrics.json"),
        os.path.join(save_path, "obj_*", "perception_renders", "metrics.json"),
    ]:
        for p in glob_module.glob(pattern):
            parent = os.path.dirname(p)
            if os.path.basename(parent) == 'perception_renders':
                parent = os.path.dirname(parent)
            if parent not in json_files:
                json_files[parent] = p
    json_files = sorted(json_files.values())

    if not json_files:
        print(f"[aggregate_from_json] No metrics.json files found in {save_path}/obj_*/")
        return 0

    print(f"\n{'='*80}")
    print(f"Aggregating from {len(json_files)} metrics.json files")
    print(f"{'='*80}\n")

    # Metric columns to extract (json_key -> csv_column)
    _METRIC_COLS = [
        ('adds_thres', 'ADD-S'), ('add_thres', 'ADD'),
        ('mean_ar', 'AR'), ('mean_vsd', 'VSD'), ('mean_mssd', 'MSSD'), ('mean_mspd', 'MSPD'),
        ('chamfer_dist', 'CHAMFER'), ('chamfer_dist_no_icp', 'CHAMFER_NO_ICP'),
        ('chamfer_dist_icp_with_scale', 'CHAMFER_ICP_WITH_SCALE'),
        ('chamfer_dist_color', 'CHAMFER_color'), ('chamfer_dist_color_no_icp', 'CHAMFER_color_NO_ICP'),
        ('chamfer_normalized', 'chamfer_normalized'),
        ('chamfer_normalized_icp_with_scale', 'chamfer_normalized_icp_with_scale'),
        ('addss', 'ADDSS'), ('addss_10', 'ADDSS_10'), ('addss_05', 'ADDSS_05'), ('addss_02', 'ADDSS_02'),
        ('err_R', 'R_error'), ('err_T', 'T_error'),
        ('lpips', 'LPIPS'), ('lpips_no_icp', 'LPIPS_NO_ICP'),
        ('ssim', 'SSIM'), ('ssim_no_icp', 'SSIM_NO_ICP'),
        ('psnr', 'PSNR'), ('psnr_no_icp', 'PSNR_NO_ICP'),
        ('color_dist', 'COLOR_DIST'), ('color_dist_no_icp', 'COLOR_DIST_NO_ICP'),
    ]
    # Binary/percentage metrics (multiply by 100 for display)
    _PCT_COLS = {'ADD-S', 'ADD', 'AR', 'VSD', 'MSSD', 'MSPD', 'ADDSS_10', 'ADDSS_05', 'ADDSS_02'}

    rows = []
    loaded = 0
    errors = 0

    for json_path in json_files:
        try:
            result = _load_metrics_from_json(json_path)
            if result is None:
                errors += 1
                continue
            object_id, metrics, file_name, failed = result
            frame_id = str(int(file_name.split('.')[0]))
            row = {'Frame_ID': frame_id, 'Class': object_id, 'FAILED': 1 if failed else 0}
            for json_key, col in _METRIC_COLS:
                val = metrics.get(json_key, np.nan)
                row[col] = val
            rows.append(row)
            loaded += 1
        except Exception as e:
            print(f"  [WARN] Failed to load {json_path}: {e}")
            errors += 1

    if loaded == 0:
        print(f"[aggregate_from_json] No valid metrics loaded (errors={errors})")
        return 0

    print(f"Loaded {loaded} samples ({errors} errors)")

    df_all = pd.DataFrame(rows)

    # Per-class summary
    class_rows = []
    for cls_id, grp in df_all.groupby('Class'):
        row = {'Class_ID': cls_id, 'N_total': len(grp), 'N_failed': int(grp['FAILED'].sum())}
        for _, col in _METRIC_COLS:
            vals = grp[col].dropna()
            if len(vals) > 0:
                mean_val = vals.mean()
                row[col] = mean_val * 100 if col in _PCT_COLS else mean_val
            else:
                row[col] = np.nan
        class_rows.append(row)

    # Overall mean
    overall = {'Class_ID': 'MEAN', 'N_total': len(df_all), 'N_failed': int(df_all['FAILED'].sum())}
    for _, col in _METRIC_COLS:
        vals = df_all[col].dropna()
        if len(vals) > 0:
            mean_val = vals.mean()
            overall[col] = mean_val * 100 if col in _PCT_COLS else mean_val
        else:
            overall[col] = np.nan
    class_rows.append(overall)
    df_classes = pd.DataFrame(class_rows)

    # All-frames with MEAN row
    mean_row = {'Frame_ID': 'MEAN', 'Class': 'ALL', 'FAILED': int(df_all['FAILED'].sum())}
    for _, col in _METRIC_COLS:
        vals = df_all[col].dropna()
        mean_row[col] = vals.mean() if len(vals) > 0 else np.nan
    df_all_with_mean = pd.concat([df_all, pd.DataFrame([mean_row])], ignore_index=True)

    # Statistics (MEAN / MEDIAN)
    stats_rows = []
    for stat_name, stat_fn in [('MEAN', 'mean'), ('MEDIAN', 'median')]:
        row = {'Statistic': stat_name}
        for _, col in _METRIC_COLS:
            vals = df_all[col].dropna()
            if len(vals) > 0:
                row[col] = getattr(vals, stat_fn)()
            else:
                row[col] = np.nan
        stats_rows.append(row)
    df_stats = pd.DataFrame(stats_rows)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    df_classes.to_csv(os.path.join(output_dir, '0_mean_all_metrics_classes_results.csv'), index=False)
    df_all_with_mean.to_csv(os.path.join(output_dir, '0_all_frames_metrics_results.csv'), index=False)
    df_stats.to_csv(os.path.join(output_dir, '0_all_frames_metrics_results_statistics.csv'), index=False)

    # Failed samples
    failed_df = df_all[df_all['FAILED'] == 1][['Class', 'Frame_ID']]
    if len(failed_df) > 0:
        failed_df.to_csv(os.path.join(output_dir, '0_failed_samples.csv'), index=False)

    print(f"Saved JSON-aggregated results to {output_dir}")
    print(f"  Per-class summary: 0_mean_all_metrics_classes_results.csv")
    print(f"  All frames: 0_all_frames_metrics_results.csv ({loaded} samples)")
    print(f"  Statistics: 0_all_frames_metrics_results_statistics.csv")

    # --- Aggregate pose 2 metrics (if any) ---
    pose2_files = sorted(glob_module.glob(os.path.join(save_path, "obj_*", "metrics_pose2.json")))
    if pose2_files:
        pose2_rows = []
        for p2_path in pose2_files:
            try:
                with open(p2_path) as f:
                    d = json.load(f)
                obj_folder = os.path.dirname(p2_path)
                obj_id = int(os.path.basename(obj_folder).split('_')[1])
                file_name = d.get('file_name', '')
                frame_id = str(int(file_name.split('.')[0])) if file_name else ''
                row = {'Frame_ID': frame_id, 'Class': obj_id, 'FAILED': 1 if d.get('_failed', False) else 0}
                for json_key, col in _METRIC_COLS:
                    val = d.get(json_key, np.nan)
                    if val is None:
                        val = np.nan
                    row[col] = val
                pose2_rows.append(row)
            except Exception:
                pass

        if pose2_rows:
            df_pose2 = pd.DataFrame(pose2_rows)
            pose2_dir = os.path.join(output_dir, 'pose2')
            os.makedirs(pose2_dir, exist_ok=True)

            # Statistics
            stats_rows_p2 = []
            for stat_name, stat_fn in [('MEAN', 'mean'), ('MEDIAN', 'median')]:
                row = {'Statistic': stat_name}
                for _, col in _METRIC_COLS:
                    vals = df_pose2[col].dropna()
                    row[col] = getattr(vals, stat_fn)() if len(vals) > 0 else np.nan
                stats_rows_p2.append(row)
            df_stats_p2 = pd.DataFrame(stats_rows_p2)

            df_pose2.to_csv(os.path.join(pose2_dir, '0_all_frames_metrics_results.csv'), index=False)
            df_stats_p2.to_csv(os.path.join(pose2_dir, '0_all_frames_metrics_results_statistics.csv'), index=False)
            print(f"\n  Pose 2 metrics: {len(pose2_rows)} samples -> {pose2_dir}")

    return loaded


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-sample metrics.json into CSVs")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Run directory containing obj_*/metrics.json (e.g. .../dataset_HB_multiview1)")
    args = parser.parse_args()
    aggregate_from_json(args.run_dir)


if __name__ == "__main__":
    main()
