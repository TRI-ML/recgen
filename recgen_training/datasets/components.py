"""
Shared preprocessing functions for the WebDataset training loaders.

The numerics here define the training preprocessing exactly (crop
augmentation, LANCZOS/NEAREST resampling, kornia depth-to-3D pointmap
computation); inference preprocessing must stay consistent with them.
"""

from typing import *
import torch
import numpy as np
from PIL import Image
import kornia as kn
import torch.nn.functional as F


def fit_unit_cube_uniform(X):
    """
    Fit point cloud X (N,D) into [0,1]^D with uniform scale only.

    Args:
        X: Point cloud array (N, D)

    Returns:
        T: 4x4 transformation matrix
        s: Scale factor
        t: Translation vector
    """
    X = np.asarray(X)
    m = X.min(axis=0)
    M = X.max(axis=0)
    size = M - m
    max_side = size.max()

    # Degenerate cloud: all points identical
    if max_side == 0:
        s = 0.0
        t = np.full(X.shape[1], 0.5)
        return s, t

    s = 1.0 / max_side

    # Map to [0,1]^D with maximal uniform scale and center leftover space
    a = (1.0 - size * s) / 2.0
    t = a - m * s

    # Make transform matrix
    T = np.eye(4)
    T[0, 0] = T[1, 1] = T[2, 2] = s
    T[:3, 3] = t

    return T, s, t


def fit_unit_cube_median_quantile(X, quantile_drop_threshold=0.025):
    """
    Fit point cloud X (N,D) into [0,1]^D using median+quantile (SAM3D-style).

    More robust to outliers than min/max approach. Uses median for center
    and quantile-based scale to exclude extreme points.

    Args:
        X: Point cloud array (N, D)
        quantile_drop_threshold: Fraction of points to drop from each tail (e.g., 0.025 = 2.5%)

    Returns:
        T: 4x4 transformation matrix
        s: Scale factor
        t: Translation vector
    """
    X = np.asarray(X)

    # Compute median for shift (robust center)
    shift = np.nanmedian(X, axis=0)

    # Shift points to center
    shifted = X - shift

    # Compute norm of each shifted point
    norms = np.linalg.norm(shifted, axis=1)

    # Compute quantiles of norms
    lower_q = np.nanquantile(norms, quantile_drop_threshold)
    upper_q = np.nanquantile(norms, 1.0 - quantile_drop_threshold)

    # Scale based on quantile range
    diameter = (upper_q - lower_q) * 2.0

    if diameter == 0:
        s = 1.0
        t = np.full(X.shape[1], 0.5) - shift
    else:
        s = 1.0 / diameter
        # Center in [0,1]^D
        t = np.full(X.shape[1], 0.5) - shift * s

    # Make transform matrix
    T = np.eye(4)
    T[0, 0] = T[1, 1] = T[2, 2] = s
    T[:3, 3] = t

    return T, s, t


# =============================================================================
# Preprocessing Functions (shared between components.py and webdataset loader)
# =============================================================================

def compute_aug_bbox(
    mask: np.ndarray,
    image_size: int,
    max_increase_ratio: float = 3.0,
    aug_size_ratio: Optional[float] = None,
) -> List[int]:
    """
    Compute augmented crop bbox centered on mask with random padding.

    Args:
        mask: Binary mask array (H, W) or PIL Image
        image_size: Target output size
        max_increase_ratio: Maximum upscale ratio to avoid too small crops
        aug_size_ratio: If None, randomly sample from [1.2, 2.0]. Otherwise use this fixed value.

    Returns:
        aug_bbox: [left, top, right, bottom] crop coordinates
    """
    if isinstance(mask, Image.Image):  # PIL Image
        img_width, img_height = mask.size
        mask_arr = np.array(mask)
    else:
        mask_arr = np.asarray(mask)
        img_height, img_width = mask_arr.shape[:2]

    bbox = mask_arr.nonzero()
    if bbox[0].size == 0 or bbox[1].size == 0:
        # Use full image as fallback when no content is found
        return [0, 0, img_width, img_height]

    bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2

    if aug_size_ratio is None:
        aug_size_ratio = np.random.uniform(1.2, 2.0)

    # Avoid too small crops
    min_size = image_size / max_increase_ratio
    min_hsize = min_size / 2
    aug_hsize = max(hsize * aug_size_ratio, min_hsize)

    aug_center = [center[0], center[1]]
    aug_bbox = [
        int(aug_center[0] - aug_hsize),
        int(aug_center[1] - aug_hsize),
        int(aug_center[0] + aug_hsize),
        int(aug_center[1] + aug_hsize)
    ]

    # Clamp to image bounds
    aug_bbox = [
        max(0, min(aug_bbox[0], img_width - 1)),
        max(0, min(aug_bbox[1], img_height - 1)),
        min(img_width, max(aug_bbox[2], aug_bbox[0] + 1)),
        min(img_height, max(aug_bbox[3], aug_bbox[1] + 1))
    ]

    return aug_bbox


def crop_and_resize_image(
    image: Image.Image,
    aug_bbox: List[int],
    target_size: int,
    resample: int = Image.Resampling.LANCZOS,
    convert_to_rgb: bool = True,
) -> torch.Tensor:
    """
    Crop image to aug_bbox and resize to target_size.

    Args:
        image: PIL Image
        aug_bbox: [left, top, right, bottom] crop coordinates
        target_size: Output size (square)
        resample: PIL resampling mode
        convert_to_rgb: Whether to convert to RGB

    Returns:
        Tensor of shape (C, H, W) normalized to [0, 1]
    """
    cropped = image.crop(aug_bbox)
    resized = cropped.resize((target_size, target_size), resample)
    if convert_to_rgb:
        resized = resized.convert('RGB')
    return torch.tensor(np.array(resized)).permute(2, 0, 1).float() / 255.0


def crop_and_resize_mask(
    mask: Image.Image,
    aug_bbox: List[int],
    target_size: int,
) -> torch.Tensor:
    """
    Crop mask to aug_bbox and resize to target_size using nearest neighbor.

    Args:
        mask: PIL Image (grayscale mask)
        aug_bbox: [left, top, right, bottom] crop coordinates
        target_size: Output size (square)

    Returns:
        Tensor of shape (H, W) normalized to [0, 1]
    """
    cropped = mask.crop(aug_bbox)
    resized = cropped.resize((target_size, target_size), Image.Resampling.NEAREST)
    return torch.tensor(np.array(resized)).float() / 255.0


def compute_pointmap_from_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    mask: torch.Tensor,
    aug_bbox: List[int],
    target_size: int,
    depth_scale: float = 1000.0,
    filter_outliers_type: Optional[str] = None,
    filter_outliers_params: Optional[Dict[str, Any]] = None,
    normalization_method: str = 'minmax',
    quantile_drop_threshold: float = 0.025,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Convert depth to normalized pointmap, crop, and resize.

    Args:
        depth: Depth tensor (H, W) in raw units (typically millimeters from PNG)
        intrinsics: Camera intrinsics matrix (3, 3)
        mask: Binary mask tensor (H, W)
        aug_bbox: [left, top, right, bottom] crop coordinates
        target_size: Output size (square)
        depth_scale: Divide depth by this value (1000 for mm->m conversion)
        filter_outliers_type: Not supported in the release (kept for config
            compatibility; the released training configs never set it).
        filter_outliers_params: Parameters for outlier removal.
        normalization_method: 'minmax' (default) or 'median_quantile' for SAM3D-style normalization.
        quantile_drop_threshold: For median_quantile, fraction to drop from each tail (e.g., 0.025 = 2.5%).

    Returns:
        Tuple of:
            - pointmap_ncam: Normalized pointmap (3, target_size, target_size)
            - cam2ncam: 4x4 camera to normalized camera transformation matrix
    """
    # Convert depth to meters
    depth_m = depth.float() / depth_scale

    # Ensure proper dimensions for kornia
    if depth_m.ndim == 2:
        depth_m = depth_m[None, None, ...]  # (1, 1, H, W)
    elif depth_m.ndim == 3:
        depth_m = depth_m[None, ...]  # (1, C, H, W)

    if intrinsics.ndim == 2:
        intrinsics = intrinsics[None, ...]  # (1, 3, 3)

    # Convert depth to 3D points
    pointmap = kn.geometry.depth.depth_to_3d(
        depth_m, intrinsics, normalize_points=False
    )[0].float()  # (3, H, W)

    # Get depth back to (H, W) for masking
    depth_hw = depth_m[0, 0]  # (H, W)

    # Ensure mask is boolean (H, W)
    if mask.ndim == 3:
        mask = mask[0]
    mask_bool = mask.bool()

    # Filter out invalid points (small norm or zero depth)
    pointmap_norm = torch.linalg.norm(pointmap, dim=0)
    valid_mask = mask_bool & (pointmap_norm > 0.05) & (depth_hw > 0)

    if valid_mask.sum() == 0:
        # Return zeros if no valid points
        return torch.zeros(3, target_size, target_size), np.eye(4)

    # Transform to normalized camera space
    X = pointmap[:, valid_mask].permute(1, 0).numpy()
    if filter_outliers_type is not None:
        raise NotImplementedError(
            "filter_outliers_type is not supported in the public release "
            "(requires open3d). The released training configs never set it."
        )

    # Use appropriate normalization method
    if normalization_method == 'median_quantile':
        cam2ncam, s, _ = fit_unit_cube_median_quantile(X, quantile_drop_threshold)
    else:  # minmax (default)
        cam2ncam, s, _ = fit_unit_cube_uniform(X)
    cam2ncam = np.array(cam2ncam)

    # Apply transformation
    pointmap_ncam = pointmap.clone()
    valid_points = pointmap[:, valid_mask].permute(1, 0)  # (N, 3)
    transformed = (valid_points @ torch.from_numpy(cam2ncam[:3, :3].T).float() +
                   torch.from_numpy(cam2ncam[:3, 3]).float())
    pointmap_ncam[:, valid_mask] = transformed.permute(1, 0).float()
    pointmap_ncam[:, ~valid_mask] = 0

    # Crop to align with image conditioning
    left, upper, right, lower = aug_bbox
    pointmap_ncam_crop = pointmap_ncam[:, upper:lower, left:right]

    # Resize using nearest neighbor interpolation
    pointmap_ncam_resized = F.interpolate(
        pointmap_ncam_crop[None],
        size=(target_size, target_size),
        mode='nearest'
    )[0]

    pointmap_ncam_resized = pointmap_ncam_resized.clamp(-2.0, 3.0)

    return pointmap_ncam_resized, cam2ncam
