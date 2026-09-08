"""RecGen mesh generation for evaluation, wired to the public
``recgen_inference`` package.

Restrictions (paper-eval configuration only):
  * ``filter_outliers_type`` must be None (paper config: null).
  * ``normalization_method`` must be ``'median_quantile'`` (paper config).
  * ``return_prefilter`` debug outputs are not supported.
"""

import os
import json

import numpy as np
import torch
import trimesh
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from PIL import Image
import imageio

from recgen_inference import build_recgen
from recgen_inference.preprocessing import (
    apply_mask_erosion,
    pointmap_from_depth,
    fit_unit_cube_median_quantile,
)
from recgen_inference.recgen_modules.utils import render_utils, postprocessing_utils
from recgen_inference.recgen_modules.utils.render_utils import render_mesh_overlay_pyrender
from recgen_inference.recgen_modules.utils.pose_utils import parse_pose_output


def load_pipeline_multiview(checkpoint_slat=None, checkpoint_sparse=None):
    """Load the RecGen pipeline with multi-view SS checkpoint.

    Thin wrapper over the public ``build_recgen.build`` (which replicates the
    internal ``load_pipeline``): pass local checkpoint paths, or None to
    auto-download the released paper checkpoints from HuggingFace
    (TRI-ML/RecGen). Model configs and pose-normalization stats are resolved
    from files next to the checkpoints, falling back to the HF repo — the same
    resolution order the paper evaluation used.
    """
    return build_recgen.build(
        "recgen_base.multiview_stereo",
        checkpoint_slat=checkpoint_slat,
        checkpoint_sparse=checkpoint_sparse,
    )


def crop_to_bounding_box(pointmap_scene, image, mask, valid, image_size=518, aug_size_ratio=1.2, max_increase_ratio=3.0, filter_outliers_type=None, filter_outliers_params=None, return_prefilter=False, normalization_method='minmax', quantile_drop_threshold=0.025, clamp_range=(-2.0, 3.0), full_image_crop=False):
    if filter_outliers_type is not None:
        raise NotImplementedError(
            "filter_outliers is not shipped in the eval release (paper config uses filter_outliers_type: null)"
        )
    if return_prefilter:
        raise NotImplementedError("return_prefilter debug outputs are not shipped in the eval release")
    if normalization_method != 'median_quantile':
        raise NotImplementedError(
            f"normalization_method={normalization_method!r} is not shipped in the eval release "
            f"(paper config uses 'median_quantile')"
        )

    bbox = np.array(mask).nonzero()
    if bbox[0].size == 0 or bbox[1].size == 0:
        width, height = mask.size  # PIL Image: .size returns (width, height)
        bbox = [0, 0, width, height]
    else:
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]

    if full_image_crop:
        # Match training's full_image_crop=true: use the entire input image as the crop window.
        width, height = image.size
        aug_bbox = [0, 0, width, height]
    else:
        # Apply fixed cropping with aug_size_ratio
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2

        # Avoid too small crops
        min_size = image_size / max_increase_ratio
        min_hsize = min_size / 2
        aug_hsize = max(hsize * aug_size_ratio, min_hsize)

        aug_bbox = [
            int(center[0] - aug_hsize),
            int(center[1] - aug_hsize),
            int(center[0] + aug_hsize),
            int(center[1] + aug_hsize)
        ]

    # Crop and resize image
    image_cropped = image.crop(aug_bbox)
    image_resized = image_cropped.resize((image_size, image_size), Image.Resampling.LANCZOS)

    # Crop and resize mask
    mask_cropped = mask.crop(aug_bbox)
    mask_resized = mask_cropped.resize((image_size, image_size), Image.Resampling.NEAREST)

    mask_tensor = torch.from_numpy(np.array(mask)).bool()  & torch.from_numpy(valid).bool()
    pointmap_3d = torch.from_numpy(pointmap_scene).float().permute(2, 0, 1)
    X = pointmap_3d[:, mask_tensor].permute(1, 0)

    # Store all valid points for normalization (before any filtering)
    X_all = X.clone()

    # Match training: use robust median+quantile normalization on all valid points,
    # and keep all valid points in the pointmap (no zeroing of outliers)
    cam2ncam, s, _ = fit_unit_cube_median_quantile(X_all.numpy(), quantile_drop_threshold)
    pointmap_ncam = pointmap_3d.clone()
    pointmap_ncam[:, mask_tensor] = (pointmap_ncam[:, mask_tensor].permute(1, 0) @ cam2ncam[:3, :3].T + cam2ncam[:3, 3]).permute(1, 0).float()
    pointmap_ncam = (pointmap_ncam * mask_tensor.int())

    left, upper, right, lower = aug_bbox
    #pointmap_ncam_crop = pointmap_ncam[:, upper:lower, left:right]
    out_h = lower - upper
    out_w = right - left
    H, W = pointmap_ncam.shape[1], pointmap_ncam.shape[2]
    pointmap_ncam_crop = torch.zeros((3, out_h, out_w), dtype=pointmap_ncam.dtype)
    src_top = max(upper, 0)
    src_bottom = min(lower, H)
    src_left = max(left, 0)
    src_right = min(right, W)

    dst_top = src_top - upper
    dst_left = src_left - left

    pointmap_ncam_crop[:, dst_top:dst_top + (src_bottom - src_top), dst_left:dst_left + (src_right - src_left)] = \
        pointmap_ncam[:, src_top:src_bottom, src_left:src_right]

    pointmap_ncam = F.interpolate(
        pointmap_ncam_crop[None],
        size=(image_size, image_size),
        mode='nearest'
    )[0].contiguous()
    pointmap = pointmap_ncam.clone()
    if clamp_range is not None:
        pointmap = pointmap.clamp(*clamp_range)

    return pointmap, image_resized, mask_resized, cam2ncam


def preprocess_view(image, depth, mask, intrinsics, filter_outliers_type=None, filter_outliers_params=None, normalization_method='minmax', quantile_drop_threshold=0.025, clamp_range=(-2.0, 3.0), mask_erosion_enabled=True, mask_erosion_params=None, aug_size_ratio=1.2, full_image_crop=False):
    """
    Preprocess a single view for multi-view inference.

    Args:
        image: RGB image (numpy array HxWx3)
        depth: Depth image (numpy array HxW)
        mask: Binary mask (numpy array HxW)
        intrinsics: Camera intrinsics (3x3 numpy array)
        filter_outliers_type: Type of outlier filtering
        filter_outliers_params: Parameters for outlier filtering
        mask_erosion_enabled: Whether to apply mask erosion
        mask_erosion_params: Parameters for mask erosion

    Returns:
        dict with 'image', 'pointmap', 'mask', 'cam2ncam' tensors
    """
    # Apply configurable mask erosion
    mask_eroded, _ = apply_mask_erosion(
        mask.copy(),
        enabled=mask_erosion_enabled,
        params=mask_erosion_params
    )

    if mask_eroded.dtype == bool:
        mask_eroded = mask_eroded.astype(np.uint8)
    if mask_eroded.max() == 1:
        mask_eroded = mask_eroded * 255

    # Filter depth with mask
    depth_masked = depth.copy()
    depth_masked[mask_eroded == 0] = 0.0
    valid = depth_masked > 0

    pil_image = Image.fromarray(image)

    # Generate pointmap from depth
    pointmap_scene = pointmap_from_depth(depth_masked, intrinsics)

    # Crop to bounding box
    pointmap_tensor, resized_image, resized_mask, cam2ncam = crop_to_bounding_box(
        pointmap_scene, pil_image, Image.fromarray(mask_eroded), valid,
        filter_outliers_type=filter_outliers_type,
        filter_outliers_params=filter_outliers_params,
        normalization_method=normalization_method,
        quantile_drop_threshold=quantile_drop_threshold,
        clamp_range=clamp_range,
        aug_size_ratio=aug_size_ratio,
        full_image_crop=full_image_crop,
    )

    return {
        'image': resized_image,
        'pointmap': pointmap_tensor,
        'mask': resized_mask,
        'cam2ncam': cam2ncam,
        'points_ncam': pointmap_tensor.permute(1, 2, 0).numpy(),  # (H, W, 3) for PLY saving
    }


def get_pose(outputs, pose_representation=None):
    """
    Extract pose transformation from model outputs.

    Args:
        outputs: Model outputs dictionary containing 'pose' tensor
        pose_representation: Pose format - 'quaternion_translation_scale', '6d_translation_scale', or '9d_translation_scale'
                           If None, automatically detects based on tensor shape

    Returns:
        T: 4x4 transformation matrix
        r: Rotation matrix (3x3) scaled by object scale
        t: Translation vector (3,)
    """
    pose_tensor = outputs['pose'][0]

    # Auto-detect pose representation from tensor shape if not provided
    if pose_representation is None:
        pose_dim = pose_tensor.shape[0]
        if pose_dim == 8:
            pose_representation = 'quaternion_translation_scale'
        elif pose_dim == 10:
            pose_representation = '6d_translation_scale'
        elif pose_dim == 13:
            pose_representation = '9d_translation_scale'
        else:
            raise ValueError(f"Unexpected pose tensor dimension: {pose_dim}. Expected 8, 10, or 13.")

    print(f"Predicted pose ({pose_representation}): {pose_tensor.detach().cpu().numpy()}")

    # Parse pose using universal parser
    parsed_pose = parse_pose_output(pose_tensor, pose_representation)

    # Extract components
    rot_matrix = parsed_pose['rotation_matrix']
    t = parsed_pose['translation']
    scale = parsed_pose['scale']

    # Apply scale to rotation matrix
    r = rot_matrix * scale

    # Apply 90-degree X rotation for GLB coordinate system
    r90 = R.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix()
    R_total = r @ r90

    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_total
    T[:3, 3] = t

    return T, r, t


def generate_mesh_multiview(pipeline, views, save_folder, obj, visualize=True, use_pointmap=True,
                            use_predicted_scale=True, save_videos=False, save_outputs_flag=True,
                            filter_outliers_type=None, filter_outliers_params=None,
                            normalization_method='minmax', quantile_drop_threshold=0.025,
                            clamp_range=(-2.0, 3.0),
                            mask_erosion_enabled=True, mask_erosion_params=None,
                            seed=1, slat_single_view=False,
                            aug_size_ratio=1.2, full_image_crop=False, use_masked_rgb=False):
    """
    Generate mesh from multiple views.

    Args:
        pipeline: RecGen pipeline loaded with the multi-view checkpoint
        views: List of view dicts, each with 'rgb', 'depth', 'mask', 'camera_intrinsics'
        save_folder: Path to save outputs
        obj: Object ID
        visualize: Whether to generate visualizations
        use_pointmap: Whether to use pointmap conditioning
        use_predicted_scale: Whether to use predicted scale
        save_videos: Whether to save video outputs
        save_outputs_flag: Whether to save detailed outputs
        filter_outliers_type: Type of outlier filtering
        filter_outliers_params: Parameters for outlier filtering

    Returns:
        mesh: Generated mesh in first view's camera frame
    """
    num_views = len(views)

    # Assertions: Verify multi-view input correctness
    assert num_views >= 1, f"Multi-view requires at least 1 view, got {num_views}"
    assert all('rgb' in v for v in views), "All views must have 'rgb' key"
    assert all('depth' in v for v in views), "All views must have 'depth' key"
    assert all('mask' in v for v in views), "All views must have 'mask' key"
    assert all('camera_intrinsics' in v for v in views), "All views must have 'camera_intrinsics' key"

    # Verify all views have same object_id
    obj_ids = [v.get('object_id') for v in views]
    if obj_ids[0] is not None:
        assert all(oid == obj_ids[0] for oid in obj_ids), (
            f"All views must have same object_id! Got: {obj_ids}"
        )

    # Verify views have different image_ids (different viewpoints)
    img_ids = [v.get('image_id') for v in views]
    if num_views > 1 and img_ids[0] is not None:
        assert len(set(img_ids)) == len(img_ids), (
            f"Multi-view must have different image_ids! Got: {img_ids}"
        )

    print(f"\n  Processing {num_views} views (img_ids: {img_ids})...")

    # Preprocess all views
    processed_views = []
    for i, view in enumerate(views):
        print(f"    Preprocessing view {i+1}/{num_views}...")
        processed = preprocess_view(
            view['rgb'], view['depth'], view['mask'], view['camera_intrinsics'],
            filter_outliers_type=filter_outliers_type,
            filter_outliers_params=filter_outliers_params,
            normalization_method=normalization_method,
            quantile_drop_threshold=quantile_drop_threshold,
            clamp_range=clamp_range,
            mask_erosion_enabled=mask_erosion_enabled,
            mask_erosion_params=mask_erosion_params,
            aug_size_ratio=aug_size_ratio,
            full_image_crop=full_image_crop,
        )
        processed_views.append(processed)

    # Assertions: Verify preprocessing produced valid outputs
    assert len(processed_views) == num_views, (
        f"Preprocessing failed: expected {num_views} views, got {len(processed_views)}"
    )
    for i, p in enumerate(processed_views):
        assert 'image' in p, f"Processed view {i} missing 'image'"
        assert 'pointmap' in p, f"Processed view {i} missing 'pointmap'"
        assert 'mask' in p, f"Processed view {i} missing 'mask'"
        assert 'cam2ncam' in p, f"Processed view {i} missing 'cam2ncam'"
        # Verify pointmap has expected shape (3, H, W)
        assert p['pointmap'].dim() == 3 and p['pointmap'].shape[0] == 3, (
            f"View {i} pointmap has unexpected shape: {p['pointmap'].shape}, expected (3, H, W)"
        )

    # Stack inputs for multi-view inference
    images = [p['image'] for p in processed_views]
    pointmaps = [p['pointmap'] for p in processed_views]
    masks = [p['mask'] for p in processed_views]

    if use_masked_rgb:
        # Match training's use_masked_rgb=true: zero RGB outside the mask before DINOv2,
        # and skip the mask embedder (model has use_mask_embedder=false).
        masked_images = []
        for img, m in zip(images, masks):
            img_arr = np.array(img)
            m_arr = np.array(m)
            if m_arr.ndim == 3:
                m_arr = m_arr[..., 0]
            keep = (m_arr > 127)[..., None]
            masked_images.append(Image.fromarray((img_arr * keep).astype(np.uint8)))
        images = masked_images
        masks = None

    # Run multi-view inference
    if use_pointmap:
        outputs = pipeline.run_pointmap_multiview(
            images=images,
            pointmaps=pointmaps,
            masks=masks,
            seed=seed,
            slat_single_view=slat_single_view,
        )
    else:
        outputs = pipeline.run_pointmap_multiview(
            images=images,
            pointmaps=None,
            masks=masks,
            seed=seed,
            slat_single_view=slat_single_view,
        )

    # Extract predicted pose (anchor)
    pose_tensor = outputs['pose'][0]
    pose_dim = pose_tensor.shape[0]

    if pose_dim == 8:
        pose_representation = 'quaternion_translation_scale'
    elif pose_dim == 10:
        pose_representation = '6d_translation_scale'
    elif pose_dim == 13:
        pose_representation = '9d_translation_scale'
    else:
        raise ValueError(f"Unexpected pose tensor dimension: {pose_dim}")

    pred_pose = parse_pose_output(pose_tensor, pose_representation)

    print(f"\n  Predicted Pose ({pose_representation}) from {num_views} views:")
    print(f"    Quaternion: [{pred_pose['quaternion'][0]:.3f}, {pred_pose['quaternion'][1]:.3f}, ...]")
    print(f"    Translation: {[f'{x:.3f}' for x in pred_pose['translation']]}")
    print(f"    Scale: {pred_pose['scale']:.3f}")

    # Extract all predicted poses (for multi-pose models)
    all_pred_poses = [pred_pose]  # pose 0 is the anchor
    all_poses_tensor = outputs.get('all_poses')
    if all_poses_tensor is not None and all_poses_tensor.ndim == 3:
        num_pose_tokens = all_poses_tensor.shape[1]
        for k in range(1, num_pose_tokens):
            pose_k_tensor = all_poses_tensor[0, k]  # (D,)
            pred_pose_k = parse_pose_output(pose_k_tensor, pose_representation)
            all_pred_poses.append(pred_pose_k)
            print(f"  Predicted Pose {k} ({pose_representation}):")
            print(f"    Quaternion: [{pred_pose_k['quaternion'][0]:.3f}, {pred_pose_k['quaternion'][1]:.3f}, ...]")
            print(f"    Translation: {[f'{x:.3f}' for x in pred_pose_k['translation']]}")
            print(f"    Scale: {pred_pose_k['scale']:.3f}")

    print("\n  Creating GLB...")
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )

    # Use first view's cam2ncam for transformation back to camera frame
    cam2ncam = processed_views[0]['cam2ncam']

    if save_outputs_flag:
        os.makedirs(save_folder, exist_ok=True)

        # Save input files for all views
        images_path = os.path.join(save_folder, "input_files")
        os.makedirs(images_path, exist_ok=True)

        for i, (view, processed) in enumerate(zip(views, processed_views)):
            pil_image = Image.fromarray(view['rgb'])
            pil_image.save(os.path.join(images_path, f"original_image_view{i}.png"))
            processed['image'].save(os.path.join(images_path, f"processed_image_view{i}.png"))
            processed['mask'].save(os.path.join(images_path, f"processed_mask_view{i}.png"))

            pointmap_np = (processed['pointmap'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pointmap_image = Image.fromarray(pointmap_np, mode='RGB')
            pointmap_image.save(os.path.join(images_path, f"processed_pointmap_view{i}.png"))

            np.save(os.path.join(images_path, f"cam2ncam_view{i}.npy"), processed['cam2ncam'])
            np.save(os.path.join(images_path, f"intrinsics_view{i}.npy"), view['camera_intrinsics'])

            # Save processed pointmap as PLY (normalized camera space)
            if 'points_ncam' in processed:
                points_ncam_full = np.array(processed['points_ncam'])  # (H, W, 3)
                colors_ncam_full = np.array(processed['image']) / 255.  # (H, W, 3)
                mask_ncam = np.linalg.norm(points_ncam_full, axis=-1) > 0
                points_ncam = points_ncam_full[mask_ncam]
                colors_ncam = colors_ncam_full[mask_ncam]
                pcd_ncam = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_ncam))
                pcd_ncam.colors = o3d.utility.Vector3dVector(colors_ncam)
                o3d.io.write_point_cloud(os.path.join(images_path, f'processed_pointmap_ncam_view{i}.ply'), pcd_ncam)

        # Save sparse structure points
        if 'coords' in outputs:
            try:
                obj_coords = outputs['coords'][:, 1:] / 64.
                pcd_sparse = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_coords.cpu().numpy()))
                o3d.io.write_point_cloud(os.path.join(save_folder, "sparse_structure_points.ply"), pcd_sparse)
            except Exception as e:
                print(f"    Warning: Could not save sparse structure: {e}")

        # Save GLB and mesh
        glb_path = os.path.join(save_folder, "output.glb")
        glb.export(glb_path)
        outputs['gaussian'][0].save_ply(os.path.join(save_folder, "gaussian.ply"))

        mesh = trimesh.load(glb_path, force='mesh')
        T, _, _ = get_pose(outputs)
        mesh.apply_transform(T)
        mesh.export(os.path.join(save_folder, "pointmap_aligned.obj"))
        mesh.apply_translation(-cam2ncam[:3, 3])
        mesh.apply_scale(1.0 / cam2ncam[0, 0])
        final_obj_path = os.path.join(save_folder, "final_mesh_recgen.obj")
        mesh.export(final_obj_path)
        mesh = trimesh.load(final_obj_path, force='mesh')

        # Save metadata
        def _serialize_pose(p):
            result = {}
            for key, value in p.items():
                if isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    result[key] = float(value)
                else:
                    result[key] = value
            return result

        metadata = {
            'pred_pose': _serialize_pose(pred_pose),
            'cam2ncam': cam2ncam.tolist(),
            'num_views': num_views,
            'all_pred_poses': [_serialize_pose(p) for p in all_pred_poses],
            'all_cam2ncams': [pv['cam2ncam'].tolist() for pv in processed_views],
        }
        with open(os.path.join(save_folder, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Render overlay visualization for first view (and optionally all views)
        if visualize:
            cam2ncam_view0 = processed_views[0]['cam2ncam']
            R0 = views[0]['R']
            t0 = views[0]['t'].flatten()
            for i, (view, processed) in enumerate(zip(views, processed_views)):
                if i == 0:
                    cam2ncam_for_render = cam2ncam_view0
                else:
                    # The predicted pose is in NCAM0 space (view 0's normalized camera space).
                    # cam2ncam_inv_view_i maps NCAM_i → cam_i, but we need NCAM0 → cam_i.
                    # Compute relative rigid transform from cam0 to cam_i using BOP object poses
                    # (R, t are object-to-camera transforms: p_cam = R @ p_obj + t).
                    Ri = view['R']
                    ti = view['t'].flatten()
                    R_rel = Ri @ R0.T
                    t_rel = ti - R_rel @ t0
                    T_rel = np.eye(4)
                    T_rel[:3, :3] = R_rel
                    T_rel[:3, 3] = t_rel
                    # cam2ncam_corrected s.t. inv(cam2ncam_corrected) = T_rel @ cam2ncam_inv_view0
                    # i.e. correctly maps NCAM0 → cam0 → cam_i
                    cam2ncam_for_render = cam2ncam_view0 @ np.linalg.inv(T_rel)
                overlay_image, mesh_only_image = render_mesh_overlay_pyrender(
                    mesh=glb,
                    pred_pose_tensor=outputs['pose'][0],
                    cam2ncam=cam2ncam_for_render,
                    intrinsics=view['camera_intrinsics'],
                    original_image_pil=Image.fromarray(view['rgb']),
                    image_size=None,
                    skip_glb_rotation=False
                )
                overlay_image.save(os.path.join(save_folder, f"overlay_render_view{i}.png"))
                mesh_only_image.save(os.path.join(save_folder, f"mesh_render_only_view{i}.png"))
            print(f"    Saved overlay renders for {num_views} views")

        if save_videos:
            try:
                print("  Rendering videos...")
                gs_video = render_utils.render_video(outputs['gaussian'][0])['color']
                imageio.mimsave(os.path.join(save_folder, "gaussian.mp4"), gs_video, fps=30)

                mesh_video = render_utils.render_video(outputs['mesh'][0])['normal']
                imageio.mimsave(os.path.join(save_folder, "mesh_normals.mp4"), mesh_video, fps=30)
                print(f"    Saved gaussian.mp4 and mesh_normals.mp4")
            except Exception as e:
                print(f"    Warning: Could not render videos: {e}")

        print(f"\n  Saved outputs to: {save_folder}")
    else:
        # Just extract mesh without saving
        T, _, _ = get_pose(outputs)
        mesh = trimesh.Trimesh(
            vertices=glb.vertices.copy(),
            faces=glb.faces.copy(),
            visual=glb.visual
        )
        mesh.apply_transform(T)
        mesh.apply_translation(-cam2ncam[:3, 3])
        mesh.apply_scale(1.0 / cam2ncam[0, 0])

    return mesh
