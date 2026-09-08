#!/usr/bin/env python3
"""
Multi-view evaluation driver for RecGen (HB / ArtVIP paper evaluations).

Usage (one command per paper-table row; see EVALUATION.md):

    python -m recgen_eval.run \
        --dataset_name HB --num_views 1 \
        --path_to_datasets /path/to/eval_data \
        --save_path_root outputs/eval \
        --parallel 1 --workers_per_gpu 2

Checkpoints default to the released paper checkpoints on HuggingFace
(TRI-ML/RecGen); pass --ckpt_path_structure/--ckpt_path_slats for local files.
The instance list defaults to the packaged fixed splits
(recgen_eval/splits/...) matching the paper protocol.
"""

import argparse
import os
import sys
import warnings

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Suppress common warnings BEFORE importing torch
warnings.filterwarnings('ignore', message='The pynvml package is deprecated')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.cuda')

os.environ['SPCONV_ALGO'] = 'native'        # Avoid int32 overflow in MaskImplicitGemm on Ampere GPUs

import numpy as np
import pandas as pd
import pickle
import trimesh
import json

from .aggregate import (
    _build_mesh_for_pose_k,
    _load_metrics_from_json,
    _save_sample_metrics_json,
    _NumpyEncoder,
    aggregate_from_json,
)
from .workers import get_gpu_ids, merge_split_results, run_parallel_workers

# Lazy-loaded to keep this module importable without the CUDA inference deps
# (the disk-backed mesh path, mesh_source=disk, never needs them).
load_pipeline_multiview = None
generate_mesh_multiview = None


def _load_recgen_pipeline_imports():
    global load_pipeline_multiview, generate_mesh_multiview
    if load_pipeline_multiview is None:
        from .generate import load_pipeline_multiview as _lp, generate_mesh_multiview as _gm
        load_pipeline_multiview = _lp
        generate_mesh_multiview = _gm


# Lazy imports for evaluation packages
load_multiview_anchor_dataset = None
Evaluator = None


def load_eval_modules():
    """Lazy load evaluation modules."""
    global load_multiview_anchor_dataset, Evaluator
    if load_multiview_anchor_dataset is None:
        from .poseval6d.dataloader import load_multiview_anchor_dataset as _load_multiview
        from .poseval6d.evaluation.evaluator import Evaluator as _Evaluator
        load_multiview_anchor_dataset = _load_multiview
        Evaluator = _Evaluator
    return load_multiview_anchor_dataset, Evaluator


_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def default_config_file():
    """Packaged evaluation config (verbatim copy of the paper's configs/eval/configs.yaml)."""
    return os.path.join(_PKG_DIR, 'configs', 'eval_configs.yaml')


def default_instance_list(dataset_name, num_views):
    """Resolve the packaged fixed-split instance list used for the paper numbers.

    Mirrors the internal launcher (run_eval_all.py): 1-view always uses
    instance_list_random.txt; 2-view uses the "simple" regime for HB and
    "random" for ArtVIP.
    """
    if dataset_name == "HB":
        name = 'instance_list_random.txt' if num_views == 1 else 'instance_list_simple.txt'
        return os.path.join(_PKG_DIR, 'splits', 'HB', name)
    if dataset_name in ("AV2", "AV_final"):
        return os.path.join(_PKG_DIR, 'splits', 'AV_final', 'instance_list_random.txt')
    raise ValueError(f"No packaged instance list for dataset {dataset_name}")


def build_parser():
    parser = argparse.ArgumentParser(description="Multi-view evaluation for recgen")
    parser.add_argument("--config_file", type=str, default=None,
                       help="Evaluator config YAML (default: packaged eval_configs.yaml)")
    parser.add_argument("--ckpt_path_slats", type=str, default=None,
                       help="Path to SLAT checkpoint. Default: auto-download released paper checkpoint from HuggingFace.")
    parser.add_argument("--ckpt_path_structure", type=str, default=None,
                       help="Path to multi-view SS checkpoint. Default: auto-download released paper checkpoint from HuggingFace.")
    parser.add_argument("--experiment_name", type=str, default="multiview_eval",
                       help="Experiment name for saving results")
    parser.add_argument("--path_to_datasets", type=str, required=True,
                       help="Path to datasets root (contains HB/ and/or AV_final/)")
    parser.add_argument("--dataset_name", type=str, default="HB",
                       help="Dataset name (HB or AV_final)")
    parser.add_argument("--num_views", type=int, default=2,
                       help="Number of views for multi-view evaluation")
    parser.add_argument("--visualize", type=int, default=1,
                       help="Generate visualizations")
    parser.add_argument("--overwrite", type=int, default=1,
                       help="Overwrite existing results")
    parser.add_argument("--save_path_root", type=str, default="./output_recgen_multiview",
                       help="Root path for saving outputs")
    parser.add_argument("--use_predicted_scale", type=int, default=1,
                       help="Use predicted scale from model")
    parser.add_argument("--use_pointmap", type=int, default=1,
                       help="Use pointmap conditioning")
    parser.add_argument("--compute_metrics", type=int, default=1,
                       help="Compute evaluation metrics")
    parser.add_argument("--split_size", type=int, default=-1,
                       help="Total number of splits for parallel evaluation")
    parser.add_argument("--split_index", type=int, default=-1,
                       help="Index of this split")
    parser.add_argument("--debug", type=int, default=0,
                       help="Debug mode (fewer samples)")
    parser.add_argument("--parallel", type=int, default=0,
                       help="Enable parallel multi-GPU evaluation")
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                        help="Number of workers per GPU (allows GPU sharing)")
    parser.add_argument("--max_samples", type=int, default=-1,
                       help="Maximum number of samples to process")
    parser.add_argument("--skip_save_outputs", type=int, default=0,
                       help="Skip saving detailed outputs")
    parser.add_argument("--save_videos", type=int, default=1,
                       help="Save videos of mesh generation")
    parser.add_argument("--instance_list", type=str, default=None,
                       help="Custom path to instance_list.txt for view pairs (default: packaged paper split)")
    parser.add_argument("--clamp_range", type=float, nargs=2, default=[-2.0, 3.0], metavar=('MIN', 'MAX'), help="Clamp pointmap values to [MIN, MAX] (default: -2.0 3.0). Use --no_clamp to disable.")
    parser.add_argument("--no_clamp", action="store_true", help="Disable pointmap clamping")
    parser.add_argument("--visualize_every", type=int, default=1,
                       help="Visualize every N-th sample (default: 1 = all)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for diffusion sampling (default: 1)")
    parser.add_argument("--slat_single_view", type=int, default=0,
                       help="Use single-view (first view only) for SLAT stage (default: 0 = multi-view)")
    parser.add_argument("--fill_missing", type=int, default=0,
                       help="Only evaluate missing samples and merge with existing results")
    parser.add_argument("--hb_test_subdir", type=str, default="val_kinect",
                       help="HB dataset test subdirectory")
    parser.add_argument("--aug_size_ratio", type=float, default=1.2,
                       help="Crop padding ratio applied around the object bbox at inference. Match the training dataset's aug_size_ratio (1.2 = tight, 2.0 = medium).")
    parser.add_argument("--full_image_crop", type=int, default=0,
                       help="If 1, use the full input image as the crop window (matches training's full_image_crop=true).")
    parser.add_argument("--use_masked_rgb", type=int, default=0,
                       help="If 1, zero RGB outside the mask before DINOv2 and skip the mask embedder (matches training's use_masked_rgb=true).")
    parser.add_argument("--mesh_source", type=str, default="recgen", choices=["recgen", "disk"],
                       help="If 'disk', skip loading the RecGen multi-view pipeline and only consume cached meshes from final_mesh_recgen.obj (re-scoring mode).")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve packaged defaults (before parallel dispatch so workers inherit them)
    if args.config_file is None:
        args.config_file = default_config_file()
    if args.instance_list is None:
        args.instance_list = default_instance_list(args.dataset_name, args.num_views)

    # Parallel orchestration mode
    if args.parallel == 1 and args.split_size <= 0:
        gpu_ids = get_gpu_ids()
        split_size = run_parallel_workers(args, gpu_ids)

        use_pointmap = bool(args.use_pointmap)
        use_predicted_scale = bool(args.use_predicted_scale)
        save_path = os.path.join(
            args.save_path_root + ("_debug" if args.debug else ""),
            args.experiment_name,
            f"pointmap_{use_pointmap}_predicted_scale_{use_predicted_scale}",
            f"dataset_{args.dataset_name}_multiview{args.num_views}"
        )
        merge_split_results(save_path, split_size, use_pointmap, use_predicted_scale, args.config_file)

        # Also aggregate from per-sample JSON files
        aggregate_from_json(save_path, args.config_file)

        print("\n" + "="*80)
        print("Parallel multi-view evaluation completed successfully!")
        print("="*80 + "\n")
        sys.exit(0)

    # Single worker mode
    path_to_datasets = args.path_to_datasets
    dataset_name = args.dataset_name
    visualize = bool(args.visualize)
    use_pointmap = bool(args.use_pointmap)
    use_predicted_scale = bool(args.use_predicted_scale)
    debug = bool(args.debug)
    num_views = args.num_views

    if debug:
        args.save_path_root = args.save_path_root + "_debug"

    save_path = os.path.join(
        args.save_path_root,
        args.experiment_name,
        f"pointmap_{use_pointmap}_predicted_scale_{use_predicted_scale}",
        f"dataset_{dataset_name}_multiview{num_views}"
    )
    split_size = args.split_size
    split_index = args.split_index

    if split_size > 1 and split_index >= 0:
        output_predictions_pickle = os.path.join(save_path, "predictions", f"predictions_{split_size}_{split_index}.pkl")
        output_metrics = os.path.join(save_path, "predictions", f"final_results_{split_size}_{split_index}")
    else:
        output_predictions_pickle = os.path.join(save_path, "predictions", "predictions_1_0.pkl")
        output_metrics = os.path.join(save_path, "predictions", "final_results")

    print(f"\n{'='*80}")
    print(f"Multi-View Evaluation: {num_views} views")
    print(f"Dataset: {dataset_name}")
    print(f"SS Checkpoint: {args.ckpt_path_structure or 'HF: TRI-ML/RecGen (released paper checkpoint)'}")
    print(f"SLAT Checkpoint: {args.ckpt_path_slats or 'HF: TRI-ML/RecGen (released paper checkpoint)'}")
    print(f"Instance list: {args.instance_list}")
    print(f"Save path: {save_path}")
    print(f"{'='*80}\n")

    # Load pipeline (skip for the disk-backed mesh path — meshes are
    # pre-generated; the SS+SLAT checkpoints aren't needed).
    if args.mesh_source == "disk":
        print("mesh_source=disk: skipping multi-view pipeline load (will only consume cached meshes)")
        pipeline = None
        # When running against pre-generated meshes we must not regenerate.
        if args.overwrite == 1:
            print("[WARN] mesh_source=disk forces overwrite=0 (cannot regenerate without a pipeline)")
            args.overwrite = 0
    else:
        print("Loading multi-view pipeline...")
        _load_recgen_pipeline_imports()
        pipeline = load_pipeline_multiview(args.ckpt_path_slats, args.ckpt_path_structure)

    os.makedirs(save_path, exist_ok=True)

    # Create progress file
    progress_file = os.path.join(save_path, f"progress_{split_size}_{split_index}.txt")
    with open(progress_file, 'w') as f:
        f.write(f"Multi-view evaluation started at {pd.Timestamp.now()}\n")
        f.write(f"Split {split_index}/{split_size}\n")
        f.write(f"Num views: {num_views}\n")
        f.write("="*80 + "\n")

    # Load evaluation modules
    load_multiview_fn, EvaluatorClass = load_eval_modules()

    # Load multi-view dataset
    anchor_dataset, dataset = load_multiview_fn(
        dataset_name=dataset_name,
        path_to_datasets=path_to_datasets,
        save_path=save_path,
        num_views=num_views,
        split_size=split_size,
        split_index=split_index,
        debug=debug,
        instance_list_path=args.instance_list,
        hb_test_subdir=getattr(args, 'hb_test_subdir', 'val_kinect'),
    )

    # Get objects in the dataset
    objects = dataset.read_all_models()
    print(f"{dataset_name} Multi-view Dataset initialized with {len(anchor_dataset)} samples.")

    evaluator = EvaluatorClass(config=args.config_file)

    # Extract filter outliers configuration from evaluator
    filter_outliers_type = evaluator.filter_outliers_type
    filter_outliers_params = evaluator.filter_outliers_params

    # Extract mask erosion configuration from evaluator
    mask_erosion_enabled = evaluator.mask_erosion_enabled
    mask_erosion_params = evaluator.mask_erosion_params
    # Disable erosion for AV datasets — masks are already tight and erosion
    # can eliminate small parts entirely, causing empty-mask failures.
    if dataset_name in ("AV2", "AV_final") and mask_erosion_enabled:
        mask_erosion_enabled = False
        print(f"Mask erosion: DISABLED for {dataset_name} dataset (overriding config)")
    else:
        print(f"Mask erosion: enabled={mask_erosion_enabled}, params={mask_erosion_params}")

    # Extract pointmap normalization configuration from evaluator
    normalization_method = evaluator.normalization_method
    quantile_drop_threshold = evaluator.quantile_drop_threshold
    print(f"Pointmap normalization: method={normalization_method}, quantile_drop_threshold={quantile_drop_threshold}")

    clamp_range = None if args.no_clamp else tuple(args.clamp_range)
    print(f"Pointmap clamp_range: {clamp_range}")

    # Fill-missing mode: force overwrite=0 so existing meshes are loaded from disk
    if args.fill_missing:
        args.overwrite = 0
        print(f"\nFill-missing mode: will load existing meshes from disk, generate only missing ones")

    # Iterate through the dataset
    results_recgen = []
    total_samples = len(anchor_dataset)
    samples_processed = 0

    for idx, (views, obj_folder) in enumerate(anchor_dataset):
        # Check max_samples limit
        if args.max_samples > 0 and samples_processed >= args.max_samples:
            print(f"\nReached max_samples limit ({args.max_samples}), stopping evaluation.")
            break

        progress_pct = (samples_processed + 1) / total_samples * 100
        samples_processed += 1

        # Use first view's object_id (all views have the same object)
        object_id = views[0]['object_id']

        gt_mesh, gt_metadata = objects[object_id]

        # Assertions: Verify multi-view correctness
        assert len(views) == num_views, (
            f"Expected {num_views} views, got {len(views)} for object {object_id}"
        )

        # All views must be for the same object
        for i, view in enumerate(views):
            assert view['object_id'] == object_id, (
                f"Object ID mismatch! View 0 has object_id={object_id}, "
                f"but view {i} has object_id={view['object_id']}"
            )

        # Views must have different image_ids (different viewpoints) - only check for multi-frame
        image_ids = [v['image_id'] for v in views]
        if num_views > 1:
            assert len(set(image_ids)) == len(image_ids), (
                f"Multi-view must have different frames! Object {object_id} has duplicate image_ids: {image_ids}"
            )

        # Print view details for verification
        view_info_str = ", ".join([f"img{v['image_id']}" for v in views])
        print(f"\n[{samples_processed}/{total_samples}] ({progress_pct:.1f}%) Object {object_id}, {num_views} views: [{view_info_str}]")

        # Skip if metrics.json already exists (restart-and-resume support)
        # Check new location first, then fall back to old perception_renders/ location
        metrics_json_path = os.path.join(obj_folder, 'metrics.json')
        if not os.path.exists(metrics_json_path):
            old_path = os.path.join(obj_folder, 'perception_renders', 'metrics.json')
            if os.path.exists(old_path):
                metrics_json_path = old_path
        if args.overwrite == 0 and os.path.exists(metrics_json_path):
            try:
                result = _load_metrics_from_json(metrics_json_path)
                if result is not None:
                    loaded_obj_id, loaded_metrics, loaded_file_name, loaded_failed = result
                    print(f"  [SKIP] metrics.json exists, loading from disk")
                    if args.compute_metrics == 1:
                        results_recgen.append((object_id, loaded_metrics, loaded_file_name))
                        evaluator.add_metrics(object_id, loaded_metrics, loaded_file_name, failed=loaded_failed)
                    with open(progress_file, 'a') as f:
                        timestamp = pd.Timestamp.now()
                        status = "SKIPPED (loaded from JSON)" + (" [FAILED]" if loaded_failed else "")
                        f.write(f"[{timestamp}] Sample {samples_processed}/{total_samples} ({progress_pct:.1f}%) - Object {object_id} - {status}\n")
                        f.flush()
                    continue
            except Exception as e:
                print(f"  [WARN] Failed to load metrics.json: {e}, will recompute")

        # Generate mesh from multiple views
        # Retry once on failure; if both attempts fail, record penalty metrics
        mesh = None
        mesh_error = None
        for attempt in range(2):
            try:
                if args.overwrite == 1 or not os.path.exists(os.path.join(obj_folder, 'final_mesh_recgen.obj')):
                    if args.mesh_source == "disk":
                        # No cached mesh and we cannot regenerate. Treat as failure.
                        raise FileNotFoundError(
                            f"mesh_source=disk but {os.path.join(obj_folder, 'final_mesh_recgen.obj')} is missing"
                        )
                    sample_visualize = visualize and (samples_processed % args.visualize_every == 1 or args.visualize_every == 1)
                    mesh = generate_mesh_multiview(
                        pipeline=pipeline,
                        views=views,
                        save_folder=obj_folder,
                        obj=object_id,
                        visualize=sample_visualize,
                        use_pointmap=use_pointmap,
                        use_predicted_scale=use_predicted_scale,
                        save_videos=args.save_videos and sample_visualize,
                        save_outputs_flag=(args.skip_save_outputs == 0),
                        filter_outliers_type=filter_outliers_type,
                        filter_outliers_params=filter_outliers_params,
                        normalization_method=normalization_method,
                        quantile_drop_threshold=quantile_drop_threshold,
                        clamp_range=clamp_range,
                        mask_erosion_enabled=mask_erosion_enabled,
                        mask_erosion_params=mask_erosion_params,
                        seed=args.seed,
                        slat_single_view=bool(args.slat_single_view),
                        aug_size_ratio=args.aug_size_ratio,
                        full_image_crop=bool(args.full_image_crop),
                        use_masked_rgb=bool(args.use_masked_rgb),
                    )
                else:
                    mesh = trimesh.load(os.path.join(obj_folder, 'final_mesh_recgen.obj'))

                # Check for degenerate mesh
                if mesh.centroid is None:
                    raise ValueError("empty/degenerate mesh (centroid is None)")

                mesh_error = None
                break  # success
            except Exception as e:
                mesh_error = str(e)
                if attempt == 0:
                    print(f"[WARN] Attempt 1 failed for object {object_id}: {e} — retrying...")
                else:
                    print(f"[ERROR] Attempt 2 failed for object {object_id}: {e} — recording as FAILED")

        # Handle permanent failure after retry
        if mesh_error is not None:
            with open(progress_file, 'a') as f:
                timestamp = pd.Timestamp.now()
                f.write(f"[{timestamp}] Sample {samples_processed}/{total_samples} ({progress_pct:.1f}%) - Object {object_id} - FAILED: {mesh_error}\n")
                f.flush()
            if args.compute_metrics == 1:
                # Record failure with penalty chamfer=20 so it's counted in statistics
                failure_metrics = {
                    "_failed": True,
                    "err_R": np.nan, "err_T": np.nan,
                    "add": np.nan, "adds": np.nan,
                    "add_thres": np.nan, "adds_thres": 0.0,
                    "mssd_err": np.nan, "mspd_err": np.nan,
                    "vsd_errs": np.nan,
                    "mean_mssd": np.nan, "mean_mspd": np.nan,
                    "mssd_rec": np.nan, "mspd_rec": np.nan,
                    "vsd_rec": np.nan,
                    "mean_vsd": np.nan, "mean_ar": np.nan,
                    "chamfer_dist": 20.0,
                    "chamfer_dist_no_icp": 20.0,
                    "chamfer_dist_icp_with_scale": 20.0,
                    "chamfer_dist_color": np.nan,
                    "lpips": np.nan, "lpips_no_icp": np.nan,
                    "ssim": np.nan, "ssim_no_icp": np.nan,
                    "psnr": np.nan, "psnr_no_icp": np.nan,
                    "color_dist": np.nan, "color_dist_no_icp": np.nan,
                }
                results_recgen.append((object_id, failure_metrics, views[0]["file_name"]))
                evaluator.add_metrics(object_id, failure_metrics, views[0]["file_name"], failed=True)
                _save_sample_metrics_json(failure_metrics, obj_folder, views[0]["file_name"], failed=True)
            else:
                # Save minimal failure marker even without metric computation
                _save_sample_metrics_json({"_failed": True}, obj_folder, views[0]["file_name"], failed=True)
            continue

        centroid = mesh.centroid
        mesh.apply_translation(-centroid)
        pose = np.eye(4)
        pose[:3, 3] = centroid

        # Compute metrics using first view's GT pose
        if args.compute_metrics == 1:
            try:
                metrics = evaluator.compute_metrics_anchor(
                    object_id,
                    pose,
                    mesh,
                    views[0]["pose"],  # First view's GT pose
                    gt_mesh,
                    gt_metadata
                )
            except Exception as _metric_err:
                print(f"  [WARN] compute_metrics_anchor failed for object {object_id}: "
                      f"{type(_metric_err).__name__}: {_metric_err} — recording as FAILED")
                _save_sample_metrics_json({"_failed": True}, obj_folder, views[0]["file_name"], failed=True,
                                          pred_pose=pose, gt_pose=views[0]["pose"],
                                          gt_metadata=gt_metadata, camera_intrinsics=views[0]['camera_intrinsics'])
                results_recgen.append((object_id, {"_failed": True}, views[0]["file_name"]))
                with open(progress_file, 'a') as f:
                    f.write(f"[{pd.Timestamp.now()}] Sample {samples_processed}/{total_samples} "
                            f"({progress_pct:.1f}%) - Object {object_id} - FAILED (metrics): {_metric_err}\n")
                    f.flush()
                continue

            # --- Pose 2 metrics (second predicted pose against second view's GT) ---
            metrics_pose2 = None
            if len(views) >= 2:
                try:
                    meta_path = os.path.join(obj_folder, 'metadata.json')
                    if os.path.exists(meta_path):
                        with open(meta_path) as _mf:
                            meta = json.load(_mf)
                        all_pred_poses = meta.get('all_pred_poses', [])
                        all_cam2ncams = meta.get('all_cam2ncams', [])
                        if len(all_pred_poses) >= 2 and len(all_cam2ncams) >= 2:
                            mesh_pose2 = _build_mesh_for_pose_k(obj_folder, all_pred_poses[1], all_cam2ncams[1])
                            if mesh_pose2 is not None:
                                centroid2 = mesh_pose2.centroid
                                mesh_pose2.apply_translation(-centroid2)
                                pose2_4x4 = np.eye(4)
                                pose2_4x4[:3, 3] = centroid2
                                metrics_pose2 = evaluator.compute_metrics_anchor(
                                    object_id,
                                    pose2_4x4,
                                    mesh_pose2,
                                    views[1]["pose"],  # Second view's GT pose
                                    gt_mesh,
                                    gt_metadata
                                )
                                print(f"  Pose2 chamfer: {metrics_pose2['chamfer_dist']:.4f} "
                                      f"(pose1: {metrics['chamfer_dist']:.4f})")
                except Exception as e:
                    print(f"  [WARN] Pose 2 metrics failed for object {object_id}: {e}")

            # Save per-sample metrics JSON for quick verification and resume
            _save_sample_metrics_json(metrics, obj_folder, views[0]["file_name"],
                                      pred_pose=pose, gt_pose=views[0]["pose"],
                                      gt_metadata=gt_metadata, camera_intrinsics=views[0]['camera_intrinsics'])
            # Save pose 2 metrics to a separate file
            if metrics_pose2 is not None:
                pose2_data = {'file_name': views[1]["file_name"], '_failed': False}
                skip_keys = {'icp_transform', 'icp_transform_with_scale', 'mssd_rec', 'mspd_rec', 'vsd_rec', 'vsd_errs', '_crop_box', '_full_icp_camera_tf'}
                for k, v in metrics_pose2.items():
                    if k.startswith('_') or k in skip_keys:
                        continue
                    if isinstance(v, (float, np.floating)) and np.isnan(v):
                        pose2_data[k] = None
                    else:
                        pose2_data[k] = v
                if gt_metadata is not None:
                    pose2_data['gt_diameter'] = float(gt_metadata['diameter'])
                with open(os.path.join(obj_folder, 'metrics_pose2.json'), 'w') as f:
                    json.dump(pose2_data, f, indent=2, cls=_NumpyEncoder)

            results_recgen.append((object_id, metrics, views[0]["file_name"]))
        else:
            # Save minimal metrics.json even without metric computation (for skip on restart)
            _save_sample_metrics_json({}, obj_folder, views[0]["file_name"])

        # Update progress file
        with open(progress_file, 'a') as f:
            timestamp = pd.Timestamp.now()
            f.write(f"[{timestamp}] Sample {samples_processed}/{total_samples} ({progress_pct:.1f}%) - Object {object_id} - COMPLETED\n")
            f.flush()

    # Update progress file with completion
    with open(progress_file, 'a') as f:
        timestamp = pd.Timestamp.now()
        f.write("="*80 + "\n")
        f.write(f"[{timestamp}] Multi-view evaluation COMPLETED - {samples_processed} samples processed\n")
        f.flush()

    # Save results
    os.makedirs(os.path.dirname(output_predictions_pickle), exist_ok=True)
    if len(results_recgen) > 0:
        with open(output_predictions_pickle, 'wb') as f:
            pickle.dump(results_recgen, f)
        print(f"Saved RECGEN multi-view predictions to {output_predictions_pickle}")

    # Compute final metrics
    if len(results_recgen) > 0:
        final_evaluator = EvaluatorClass(config=args.config_file)
        for object_id, metrics, file_name in results_recgen:
            failed = metrics.get('_failed', False)
            final_evaluator.add_metrics(object_id, metrics, file_name, failed=failed)

        final_evaluator.save_metrics(output_path=output_metrics)
        print(f"Saved final RECGEN multi-view results to {output_metrics}")

    # Aggregate from per-sample JSON files (pickle-free, enables resume/re-aggregation)
    aggregate_from_json(save_path, args.config_file)

    print("\n" + "="*80)
    print(f"Multi-view evaluation completed! Processed {samples_processed} samples.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
