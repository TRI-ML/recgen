"""
WebDataset implementation for sparse structure latent training/validation.

Uses WebDataset tar shards as the storage backend for efficient multi-GPU data loading.
Preprocessing (crop/resize/pointmap) is done at runtime using shared functions from components.py.
"""

import io
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Union

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from PIL import Image
from torch.utils.data import IterableDataset

from .components import (
    compute_aug_bbox,
    crop_and_resize_image,
    crop_and_resize_mask,
    compute_pointmap_from_depth,
)
from .pose_stats import FILTER_DEFAULTS


def _load_image_as_array(data: bytes) -> np.ndarray:
    """Load PNG/JPG bytes into a numpy array (HWC, uint8)."""
    return np.array(Image.open(io.BytesIO(data)))


def _load_depth_png16(data: bytes) -> np.ndarray:
    """Load 16-bit PNG depth image as numpy array (HW, uint16)."""
    img = Image.open(io.BytesIO(data))
    return np.array(img, dtype=np.uint16)


def _load_npy(data: bytes) -> np.ndarray:
    """Load npy bytes into a numpy array."""
    return np.load(io.BytesIO(data))


# WebDataset error handler that logs instead of silently swallowing.
# Module-level (not a closure) so the dataset stays picklable for
# spawned DataLoader workers; the counter is per-process.
_wds_error_count = [0]


def _warn_and_count(exn):
    _wds_error_count[0] += 1
    if _wds_error_count[0] <= 10 or _wds_error_count[0] % 50 == 0:
        logger.warning(f"[WebDataset] error #{_wds_error_count[0]} (pid={os.getpid()}): {exn!r}")
    return True


# Forward declaration - WebDatasetImageConditionedSparseStructureLatent is defined after
# WebDatasetMultiViewSparseStructureLatent as a convenience wrapper with num_views=1


class WebDatasetMultiViewSparseStructureLatent(IterableDataset):
    """
    Unified WebDataset-backed dataset for sparse structure latent training.

    Supports both single-view and multi-view conditioning. Each WebDataset sample
    contains ALL valid views, and N views are randomly selected at runtime.

    When num_views=1, outputs match single-view API (no view dimension).
    When num_views>1, outputs have view dimension: (num_views, C, H, W).

    Expected sample keys per WebDataset sample (efficient per-file format):
        - ss_latent.npy: (8, 16, 16, 16) latent
        - cond_image_00.jpg/.png, cond_image_01.jpg/.png, ...: per-view images
        - cond_image_r_00.jpg/.png, ... (optional): per-view rendered images
        - cond_mask_00.png, cond_mask_01.png, ...: per-view masks
        - cond_mask_full_00.png, cond_mask_full_01.png, ...: per-view full masks
        - cond_depth_00.png, cond_depth_01.png, ...: per-view 16-bit depth PNGs
        - num_views.json: int, number of available views
        - view_indices.json: list of original view indices
        - pose_data.json: list of pose dicts (one per view)
        - view_metadata.json: list of metadata dicts (one per view)
        - sha256.txt, aesthetic_score.json, captions.txt, etc.
    """

    def __init__(
        self,
        *args: Any,
        shards: Optional[Union[str, Iterable[str]]] = None,
        num_views: int = 2,  # Number of views to sample per training step
        image_size: int = 518,
        normalization: Optional[dict] = None,
        use_pose_normalization: bool = False,
        pose_normalization_config: Optional[str] = None,
        pose_variant: str = "median_quantile_5per",  # Pose normalization variant to use
        min_aesthetic_score: float = 0.0,
        min_visible_fraction: float = 0.2,
        min_visible_pixels: int = 400,
        max_pose_norm_threshold: float = 10.0,
        min_scale: Optional[float] = None,
        max_scale: Optional[float] = None,
        max_translation_norm: Optional[float] = None,
        mix_stereo_depth: bool = False,
        aug_size_ratio: Optional[float] = None,
        full_image_crop: bool = False,
        color_jitter: Optional[dict] = None,  # torchvision ColorJitter kwargs, e.g.
                                              # {"brightness": 0.4, "contrast": 0.4,
                                              #  "saturation": 0.4, "hue": 0.1};
                                              # same params applied to all views of a sample.
        mask_source: str = "gt",  # "gt" | "sam2" | "mix": which mask conditions the model.
                                  # "sam2" uses cond_mask_sam2_XX.png (GT fallback); "mix" picks
                                  # SAM2 with probability sam2_mask_prob per sample when available.
        sam2_mask_prob: float = 0.5,  # only used with mask_source="mix"
        # Per-view SAM2 gate on the frame difference between the SAM2- and
        # GT-mask normalizations (computed from the two stored pose files, no
        # mask decoding needed). A view uses the SAM2 mask+pose only when
        # |log2(scale_sam2/scale_gt)| <= sam2_max_log2_scale and
        # ||t_sam2 - t_gt|| <= sam2_max_dt (cube units); otherwise it falls
        # back to GT mask + GT-frame pose.
        sam2_max_log2_scale: float = 0.07,
        sam2_max_dt: float = 0.03,
        shuffle: bool = True,
        shuffle_buffer: int = 1024,
        shardshuffle: Optional[Union[bool, int]] = None,
        interleave_shards: bool = True,
        seed: int = 9176,
        length: Optional[int] = None,
        require_pose_data: bool = True,
        require_pointmap: bool = False,
        pose_stats_file: Optional[str] = None,
        blacklist_file: Optional[str] = None,
        metadata_root: Optional[str] = None,
        **_: Any,
    ):
        """
        Args:
            shards: Shard pattern or list of shard URLs
            num_views: Number of views to randomly sample per training step.
                       Use 1 for single-view training (outputs without view dimension).
                       Use 2+ for multi-view training (outputs with view dimension).
            image_size: Target image size after preprocessing
            shuffle: Whether to shuffle samples (default True)
            shuffle_buffer: Size of shuffle buffer for samples (default 1024)
            shardshuffle: Shard-level shuffling. If None, defaults to 100 (training) or False (validation).
            use_pose_normalization: Whether to apply pose normalization (default False)
            pose_normalization_config: Path to pose normalization statistics JSON file
            pose_stats_file: Path to a pre-aggregated pose statistics JSON file
                (e.g. the released stereo_pose_stats.json / slat_pose_stats.json).
                When set, statistics are loaded directly from this file instead of
                being aggregated from per-dataset metadata next to the shards.
            pose_variant: Which pose normalization variant to use. Options:
                - "minmax": Standard min/max bounding box normalization (default)
                - "median_quantile_2.5per": SAM3D-style robust normalization (2.5% quantile)
                - "median_quantile_5per": SAM3D-style robust normalization (5% quantile)
                - "median_quantile_10per": SAM3D-style robust normalization (10% quantile)

            mix_stereo_depth: When True, randomly selects stereo or normal depth with p=0.5
                for samples that have both available. Samples with only one type use that type.

        Quality filtering parameters (if None, auto-detected from depth type):
            min_visible_fraction: Minimum visible fraction of object (default 0.2)
            min_visible_pixels: Minimum number of mask pixels (default 400)
            max_pose_norm_threshold: Maximum L2 norm of full pose vector (default 10.0)
            min_scale: Minimum valid scale (default: 0.3)
            max_scale: Maximum valid scale (auto: 3.0)
            max_translation_norm: Maximum L2 norm of translation (auto: 4.0)
        """
        super().__init__()
        if shards is None and len(args) > 0:
            shards = args[0]
        self.shards = shards
        self.num_views = num_views
        self.image_size = image_size
        self.normalization = normalization
        self.use_pose_normalization = use_pose_normalization
        self.pose_normalization_config = pose_normalization_config
        self.pose_variant = pose_variant
        self.mix_stereo_depth = mix_stereo_depth
        self.aug_size_ratio = aug_size_ratio
        self.full_image_crop = full_image_crop
        self.color_jitter = color_jitter
        self._color_jitter_tf = None
        if color_jitter:
            from torchvision import transforms as tv_transforms
            self._color_jitter_tf = tv_transforms.ColorJitter(**color_jitter)
        assert mask_source in ("gt", "sam2", "mix"), f"invalid mask_source: {mask_source}"
        self.mask_source = mask_source
        self.sam2_mask_prob = sam2_mask_prob
        self.sam2_max_log2_scale = sam2_max_log2_scale
        self.sam2_max_dt = sam2_max_dt

        # Single filter parameter dict for pose quality filtering.
        self.filter = dict(FILTER_DEFAULTS)
        if min_scale is not None:
            self.filter['min_scale'] = min_scale
        if max_scale is not None:
            self.filter['max_scale'] = max_scale
        if max_translation_norm is not None:
            self.filter['max_translation_norm'] = max_translation_norm
        if max_pose_norm_threshold is not None:
            self.filter['max_pose_norm_threshold'] = max_pose_norm_threshold

        # Legacy single-value attributes
        self.min_scale = self.filter['min_scale']
        self.max_scale = self.filter['max_scale']
        self.max_translation_norm = self.filter['max_translation_norm']
        self.max_pose_norm_threshold = self.filter['max_pose_norm_threshold']

        # Determine pose data keys based on variant
        # For backward compatibility, "minmax" uses "pose_data.json"
        if pose_variant == "minmax":
            self.pose_data_key = "pose_data.json"
            self.pose_data_key_fallback = "pose_data_minmax.json"
        else:
            self.pose_data_key = f"pose_data_{pose_variant}.json"
            self.pose_data_key_fallback = "pose_data.json"

        # SAM2-frame pose keys (pose_data_sam2_*.json, written by
        # scripts/add_sam2_poses.py). Used per sample when the SAM2 mask is
        # selected as the conditioning mask, so labels stay in the same
        # normalization frame as the inputs.
        self.pose_data_key_sam2 = (
            None if pose_variant == "minmax"
            else f"pose_data_sam2_{pose_variant}.json")

        # Stereo depth pose key (derived from base variant)
        if self.mix_stereo_depth:
            base_variant = pose_variant[3:] if pose_variant.startswith('st_') else pose_variant
            st_variant = f"st_{base_variant}"
            if base_variant == "minmax":
                self.pose_data_key_st = "pose_data_st_minmax.json"
                self.pose_data_key_st_sam2 = None
            else:
                self.pose_data_key_st = f"pose_data_{st_variant}.json"
                self.pose_data_key_st_sam2 = f"pose_data_sam2_{st_variant}.json"
        else:
            self.pose_data_key_st = None
            self.pose_data_key_st_sam2 = None

        self.min_aesthetic_score = min_aesthetic_score
        self.min_visible_fraction = min_visible_fraction
        self.min_visible_pixels = min_visible_pixels
        self.require_pose_data = require_pose_data
        self.require_pointmap = require_pointmap
        self.value_range = (0, 1)
        self.length_override = length
        self.pose_stats_file = pose_stats_file
        self.metadata_root = metadata_root

        # Resolve dataset dirs for metadata discovery (pose stats, blacklists).
        # When metadata_root is set (e.g., S3 metadata synced to local cache),
        # use it instead of parsing the shard pattern (which may be a pipe URL).
        self._metadata_dirs = None
        if self.metadata_root is not None:
            from .pose_stats import extract_dataset_names_from_shards
            from pathlib import Path
            names = extract_dataset_names_from_shards(self.shards)
            self._metadata_dirs = [Path(self.metadata_root) / n for n in names]

        # Initialize pose normalizer if enabled
        self.pose_normalizer = None
        self.pose_stats = None
        if self.use_pose_normalization:
            # Load cached stats or compute if not available
            self.pose_stats = self._load_or_compute_pose_stats()

            # Initialize pose normalizer from stats dict
            from recgen_inference.recgen_modules.utils.pose_utils import PoseNormalizer
            self.pose_normalizer = PoseNormalizer(self.pose_stats)

        # Load blacklists: auto-discover from dataset metadata + optional extra file
        self.blacklisted_shas = set()

        # Auto-load blacklist_alignment.txt from each dataset's additional_metadata/
        if self._metadata_dirs is not None:
            metadata_dirs = self._metadata_dirs
        else:
            from .pose_stats import _extract_dataset_dirs
            metadata_dirs = _extract_dataset_dirs(self.shards)
        for dataset_dir in metadata_dirs:
            bl_path = dataset_dir / "additional_metadata" / "blacklist_alignment.txt"
            if bl_path.exists():
                with open(bl_path) as f:
                    self.blacklisted_shas.update(line.strip() for line in f if line.strip())

        # Also load extra blacklist file if provided (for manual additions)
        if blacklist_file is not None:
            with open(blacklist_file) as f:
                self.blacklisted_shas.update(line.strip() for line in f if line.strip())

        if not self.blacklisted_shas:
            self.blacklisted_shas = None  # keep None semantics for no-blacklist path

        if shardshuffle is None:
            shardshuffle = True if shuffle else False
        self.shardshuffle = shardshuffle
        self.interleave_shards = interleave_shards

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization["mean"]).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization["std"]).reshape(-1, 1, 1, 1)

        self.dataset = self._build_dataset(shuffle=shuffle, shuffle_buffer=shuffle_buffer, seed=seed)

        if length is not None:
            self.loads = [1 for _ in range(length)]

    def _load_or_compute_pose_stats(self) -> dict:
        """Load cached pose stats or compute if not available.

        When pose_stats_file is set, statistics are loaded directly from that file
        (pre-aggregated format, e.g. the released *_pose_stats.json files).

        When mix_stereo_depth is True, loads stats for both normal and stereo depth
        variants and merges them with data-driven weights based on stereo availability.
        """
        if self.pose_stats_file is not None:
            with open(self.pose_stats_file) as f:
                stats = json.load(f)
            logger.info("Loaded pose normalization stats from %s", self.pose_stats_file)
            return stats
        if self.mix_stereo_depth:
            from .pose_stats import load_weighted_pose_stats
            return load_weighted_pose_stats(
                shards_pattern=self.shards,
                pose_variant=self.pose_variant,
                num_workers=8,
                verbose=True,
                dataset_dirs=self._metadata_dirs,
            )
        else:
            from .pose_stats import get_or_compute_pose_stats
            return get_or_compute_pose_stats(
                shards_pattern=self.shards,
                pose_variant=self.pose_variant,
                num_workers=8,
                verbose=True,
                dataset_dirs=self._metadata_dirs,
            )

    def _decode_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode a multi-view WebDataset sample.

        Supports efficient per-file format where each view's data is stored separately:
        - cond_image_00.jpg/.png, cond_image_01.jpg/.png, ...
        - cond_mask_00.png, cond_mask_01.png, ...
        - cond_depth_00.png, cond_depth_01.png, ...

        Randomly selects num_views from all available views and processes them.
        """
        import warnings
        import re
        import random as _random

        # Filter by blacklist if enabled
        if self.blacklisted_shas is not None:
            sha = sample.get("sha256.txt", b"").decode("utf-8") if isinstance(sample.get("sha256.txt"), bytes) else ""
            if sha and sha in self.blacklisted_shas:
                return None

        # Load latent
        sample_key = sample.get("__key__", "<unknown>")
        if "ss_latent.npy" not in sample:
            logger.warning("Skipping sample %s: missing 'ss_latent.npy'", sample_key)
            return None
        try:
            ss_latent = torch.from_numpy(_load_npy(sample["ss_latent.npy"])).float()
        except Exception as e:
            logger.warning("Skipping sample %s: failed to load ss_latent.npy: %s", sample_key, e)
            return None
        if ss_latent.shape != (8, 16, 16, 16):
            logger.warning("Skipping sample %s: unexpected latent shape %s", sample_key, ss_latent.shape)
            return None

        # Per-sample mask-source selection. The chosen mask defines the
        # normalization frame, so the pose keys must match it: SAM2 mask ->
        # pose_data_sam2_* (with GT fallback for samples lacking SAM2 data,
        # e.g. part datasets).
        use_sam2_mask = False
        if self.mask_source in ("sam2", "mix"):
            has_sam2 = any(k.startswith("cond_mask_sam2_") for k in sample)
            if has_sam2:
                use_sam2_mask = (self.mask_source == "sam2"
                                 or _random.random() < self.sam2_mask_prob)

        # Per-sample depth type selection (keys resolved for the chosen mask)
        pose_data_key_st = self.pose_data_key_st
        if (use_sam2_mask and self.pose_data_key_st_sam2 is not None
                and self.pose_data_key_st_sam2 in sample):
            pose_data_key_st = self.pose_data_key_st_sam2
        has_stereo = (self.mix_stereo_depth and
                      pose_data_key_st is not None and
                      pose_data_key_st in sample)

        if has_stereo:
            use_stereo_depth = _random.random() < 0.5  # 50/50 when both available
        else:
            use_stereo_depth = False

        # Select pose data key and filter thresholds based on depth type
        if use_stereo_depth:
            pose_data_key = pose_data_key_st
        else:
            pose_data_key = self.pose_data_key
            if use_sam2_mask and self.pose_data_key_sam2 is not None \
                    and self.pose_data_key_sam2 in sample:
                pose_data_key = self.pose_data_key_sam2
            elif pose_data_key not in sample:
                if self.pose_data_key_fallback in sample:
                    pose_data_key = self.pose_data_key_fallback
                else:
                    pose_data_key = "pose_data.json"
                    warnings.warn(f"No pose data found in {pose_data_key}, using fallback {self.pose_data_key_fallback}")
        filt = self.filter

        pose_data_list = json.loads(sample[pose_data_key].decode("utf-8"))

        # GT-frame pose list for per-view SAM2 gating fallback (same depth type)
        pose_data_list_gt = None
        if use_sam2_mask and pose_data_key.startswith("pose_data_sam2_"):
            gt_key = pose_data_key.replace("pose_data_sam2_", "pose_data_")
            if gt_key in sample:
                pose_data_list_gt = json.loads(sample[gt_key].decode("utf-8"))
        view_metadata_list = json.loads(sample["view_metadata.json"].decode("utf-8"))
        view_indices = json.loads(sample["view_indices.json"].decode("utf-8"))

        # Detect available views from sample keys (efficient per-file format)
        # Pattern: cond_image_XX.jpg or cond_image_XX.png
        image_pattern = re.compile(r"cond_image_(\d+)\.(jpg|png)$")
        available_view_nums = set()
        for key in sample.keys():
            match = image_pattern.match(key)
            if match:
                available_view_nums.add(int(match.group(1)))

        total_views = len(available_view_nums)
        if total_views == 0:
            return None

        # Sort view numbers for consistent ordering
        sorted_view_nums = sorted(available_view_nums)

        # Build per-view data arrays by loading individual files
        cond_images = []
        cond_masks = []
        cond_masks_full = []
        cond_depths = []
        cond_depths_st = []
        cond_indoors = []

        for view_num in sorted_view_nums:
            # Load image (always JPEG)
            image_key = f"cond_image_{view_num:02d}.jpg"
            if image_key not in sample:
                continue

            # Load image
            img_arr = _load_image_as_array(sample[image_key])  # (H, W, 3) or (H, W, 4)
            if img_arr.ndim == 2:
                img_arr = np.stack([img_arr] * 3, axis=-1)
            elif img_arr.shape[-1] == 4:
                img_arr = img_arr[..., :3]
            cond_images.append(img_arr)

            # Load mask. With SAM2 selected, gate per view on the FRAME
            # DIFFERENCE between the SAM2 and GT normalizations, read directly
            # from the two pose files (scale ratio + translation shift). A
            # gated-out view falls back to GT mask + GT-frame pose so the
            # normalization frame and labels stay consistent.
            local_idx_now = len(cond_masks)
            mask_key = f"cond_mask_{view_num:02d}.png"
            mask_arr = None
            if use_sam2_mask:
                sam2_key = f"cond_mask_sam2_{view_num:02d}.png"
                view_sam2_ok = False
                if (sam2_key in sample and pose_data_list_gt is not None
                        and local_idx_now < min(len(pose_data_list), len(pose_data_list_gt))):
                    e_s2 = pose_data_list[local_idx_now]
                    e_gt = pose_data_list_gt[local_idx_now]
                    if (e_s2 and e_gt and e_s2.get("sam2_converted", False)
                            and e_gt.get("scale") is not None and e_gt["scale"] > 0):
                        log2_r = abs(np.log2(e_s2["scale"] / e_gt["scale"]))
                        dt = np.sqrt(sum((e_s2[k] - e_gt[k]) ** 2
                                         for k in ("trans_x", "trans_y", "trans_z")))
                        view_sam2_ok = (log2_r <= self.sam2_max_log2_scale
                                        and dt <= self.sam2_max_dt)
                if view_sam2_ok:
                    mask_arr = _load_image_as_array(sample[sam2_key])
                    if mask_arr.ndim == 3:
                        mask_arr = mask_arr[..., 0]
                elif (pose_data_list_gt is not None
                        and local_idx_now < len(pose_data_list_gt)):
                    pose_data_list[local_idx_now] = pose_data_list_gt[local_idx_now]
            if mask_arr is None:
                if mask_key in sample:
                    mask_arr = _load_image_as_array(sample[mask_key])
                    if mask_arr.ndim == 3:
                        mask_arr = mask_arr[..., 0]
                else:
                    mask_arr = np.zeros(img_arr.shape[:2], dtype=np.uint8)
            cond_masks.append(mask_arr)

            # Load mask_full
            mask_full_key = f"cond_mask_full_{view_num:02d}.png"
            if mask_full_key in sample:
                mask_full_arr = _load_image_as_array(sample[mask_full_key])
                if mask_full_arr.ndim == 3:
                    mask_full_arr = mask_full_arr[..., 0]
                cond_masks_full.append(mask_full_arr)
            else:
                cond_masks_full.append(cond_masks[-1].copy())

            # Load depth (16-bit PNG)
            depth_key = f"cond_depth_{view_num:02d}.png"
            if depth_key in sample:
                depth_arr = _load_depth_png16(sample[depth_key])
                cond_depths.append(depth_arr.astype(np.float32))
            else:
                cond_depths.append(np.zeros(img_arr.shape[:2], dtype=np.float32))

            # Load stereo depth (16-bit PNG) — optional
            depth_st_key = f"cond_depth_st_{view_num:02d}.png"
            if depth_st_key in sample:
                cond_depths_st.append(_load_depth_png16(sample[depth_st_key]).astype(np.float32))
            else:
                cond_depths_st.append(None)

            # Load indoor background (JPEG) — optional
            indoors_key = f"cond_indoors_{view_num:02d}.jpg"
            if indoors_key in sample:
                indoors_arr = _load_image_as_array(sample[indoors_key])
                if indoors_arr.ndim == 2:
                    indoors_arr = np.stack([indoors_arr] * 3, axis=-1)
                elif indoors_arr.shape[-1] == 4:
                    indoors_arr = indoors_arr[..., :3]
                cond_indoors.append(indoors_arr)
            else:
                cond_indoors.append(None)

        # Convert to numpy arrays
        cond_images = np.stack(cond_images, axis=0)  # (N, H, W, 3)
        cond_masks = np.stack(cond_masks, axis=0)  # (N, H, W)
        cond_masks_full = np.stack(cond_masks_full, axis=0)  # (N, H, W)
        cond_depths = np.stack(cond_depths, axis=0)  # (N, H, W)
        total_views = cond_images.shape[0]

        # Filter views by quality criteria
        valid_view_local_indices = []
        for local_idx in range(total_views):
            # Check visible fraction from metadata
            if view_metadata_list[local_idx]:
                try:
                    metadata = json.loads(view_metadata_list[local_idx]) if isinstance(view_metadata_list[local_idx], str) else view_metadata_list[local_idx]
                    visible_fraction = metadata.get("visible_fraction", 1.0)
                    if visible_fraction < self.min_visible_fraction:
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass

            # Check mask pixel count
            mask_arr = cond_masks[local_idx]
            mask_pixel_count = np.count_nonzero(mask_arr)
            if mask_pixel_count <= self.min_visible_pixels:
                continue

            # Check pose quality using depth-type-specific thresholds
            pose_dict = pose_data_list[local_idx]
            try:
                if pose_dict is None:
                    continue

                quat = np.array([
                    pose_dict["quat_w"],
                    pose_dict["quat_x"],
                    pose_dict["quat_y"],
                    pose_dict["quat_z"],
                ])
                translation = np.array([
                    pose_dict["trans_x"],
                    pose_dict["trans_y"],
                    pose_dict["trans_z"],
                ])
                scale = pose_dict["scale"]

                pose_full = np.concatenate([quat, translation, [scale]])
                pose_norm = np.linalg.norm(pose_full)
                if pose_norm > filt['max_pose_norm_threshold']:
                    continue

                if scale <= filt['min_scale'] or scale >= filt['max_scale']:
                    continue

                translation_norm = np.linalg.norm(translation)
                if translation_norm > filt['max_translation_norm']:
                    continue

            except (KeyError, TypeError):
                continue

            valid_view_local_indices.append(local_idx)

        # Skip sample if not enough valid views
        if len(valid_view_local_indices) < self.num_views:
            return None

        # Randomly sample num_views from valid views
        selected_indices = np.random.choice(
            valid_view_local_indices,
            size=self.num_views,
            replace=False
        )

        # Process selected views
        processed_conds = []
        processed_masks = []
        processed_pointmaps = []
        processed_poses = []
        processed_cam2ncams = []
        processed_indoors = []
        processed_view_metadata = []
        processed_aug_bboxes = []

        # Sample color jitter params once per sample so all views get the same
        # photometric transform.
        jitter_params = None
        if self._color_jitter_tf is not None:
            from torchvision.transforms import ColorJitter
            jitter_params = ColorJitter.get_params(
                self._color_jitter_tf.brightness,
                self._color_jitter_tf.contrast,
                self._color_jitter_tf.saturation,
                self._color_jitter_tf.hue,
            )

        for local_idx in selected_indices:
            # Get raw data for this view
            image_arr = cond_images[local_idx]  # (H, W, 3)
            mask_arr = cond_masks[local_idx]  # (H, W)
            pose_dict = pose_data_list[local_idx]
            view_metadata = view_metadata_list[local_idx]

            # Select depth source based on depth type
            if use_stereo_depth:
                if cond_depths_st[local_idx] is None:
                    return None  # stereo depth missing for this view, skip sample
                depth_arr = cond_depths_st[local_idx]
            else:
                depth_arr = cond_depths[local_idx]  # (H, W)

            # Convert to PIL for preprocessing
            image_pil = Image.fromarray(image_arr)
            mask_pil = Image.fromarray(mask_arr)

            # Compute augmented crop bbox
            if self.full_image_crop:
                h, w = mask_arr.shape[:2]
                aug_bbox = [0, 0, w, h]
            else:
                aug_bbox = compute_aug_bbox(
                    mask=mask_arr,
                    image_size=self.image_size,
                    max_increase_ratio=3.0,
                    aug_size_ratio=self.aug_size_ratio,
                )

            # Crop and resize image
            cond = crop_and_resize_image(
                image=image_pil,
                aug_bbox=aug_bbox,
                target_size=self.image_size,
            )

            if jitter_params is not None:
                cond = self._apply_color_jitter(cond, jitter_params)

            # Crop and resize mask
            cond_mask = crop_and_resize_mask(
                mask=mask_pil,
                aug_bbox=aug_bbox,
                target_size=self.image_size,
            ).unsqueeze(0)  # (1, H, W)

            processed_conds.append(cond)
            processed_masks.append(cond_mask)

            # Compute pointmap from the selected depth source
            cond_pointmap = None
            cam2ncam = None
            if view_metadata:
                try:
                    metadata = json.loads(view_metadata) if isinstance(view_metadata, str) else view_metadata
                    if "intrinsics" in metadata:
                        intrinsics = torch.tensor(metadata["intrinsics"]).float()
                        depth = torch.from_numpy(depth_arr).float()
                        mask = torch.from_numpy(mask_arr).float() / 255.0

                        # Determine normalization method from pose_variant
                        # Strip st_ prefix for stereo depth variants
                        variant_for_norm = self.pose_variant
                        if variant_for_norm.startswith('st_'):
                            variant_for_norm = variant_for_norm[3:]

                        if variant_for_norm.startswith('median_quantile_'):
                            norm_method = 'median_quantile'
                            thresh_str = variant_for_norm.replace('median_quantile_', '').replace('per', '')
                            quantile_thresh = float(thresh_str) / 100
                        else:
                            norm_method = 'minmax'
                            quantile_thresh = 0.025  # unused for minmax

                        cond_pointmap, cam2ncam = compute_pointmap_from_depth(
                            depth=depth,
                            intrinsics=intrinsics,
                            mask=mask,
                            aug_bbox=aug_bbox,
                            target_size=self.image_size,
                            depth_scale=1000.0,
                            normalization_method=norm_method,
                            quantile_drop_threshold=quantile_thresh,
                        )

                        if cond_pointmap.abs().sum() == 0:
                            cond_pointmap = None
                            cam2ncam = None
                except Exception:
                    pass

            processed_pointmaps.append(cond_pointmap)
            processed_cam2ncams.append(cam2ncam)

            # Indoor background (same crop/resize as main image)
            cond_indoors_processed = None
            if cond_indoors[local_idx] is not None:
                indoors_pil = Image.fromarray(cond_indoors[local_idx])
                cond_indoors_processed = crop_and_resize_image(
                    image=indoors_pil,
                    aug_bbox=aug_bbox,
                    target_size=self.image_size,
                )
            processed_indoors.append(cond_indoors_processed)

            # Store view_metadata as JSON string and aug_bbox as tensor for run_snapshot
            processed_view_metadata.append(
                json.dumps(view_metadata) if isinstance(view_metadata, dict) else view_metadata
            )
            processed_aug_bboxes.append(torch.tensor(aug_bbox, dtype=torch.float32))

            # Extract pose tensors
            pose_8d = torch.tensor([
                pose_dict["quat_x"],
                pose_dict["quat_y"],
                pose_dict["quat_z"],
                pose_dict["quat_w"],
                pose_dict["trans_x"],
                pose_dict["trans_y"],
                pose_dict["trans_z"],
                pose_dict["scale"],
            ], dtype=torch.float32)

            rot_6d = [
                pose_dict["rot6d_0"], pose_dict["rot6d_1"], pose_dict["rot6d_2"],
                pose_dict["rot6d_3"], pose_dict["rot6d_4"], pose_dict["rot6d_5"],
            ]
            trans = [pose_dict["trans_x"], pose_dict["trans_y"], pose_dict["trans_z"]]
            scale = pose_dict["scale"]
            pose_10d = torch.tensor(rot_6d + trans + [scale], dtype=torch.float32)

            rot_9d = [
                pose_dict["rot9d_0"], pose_dict["rot9d_1"], pose_dict["rot9d_2"],
                pose_dict["rot9d_3"], pose_dict["rot9d_4"], pose_dict["rot9d_5"],
                pose_dict["rot9d_6"], pose_dict["rot9d_7"], pose_dict["rot9d_8"],
            ]
            pose_13d = torch.tensor(rot_9d + trans + [scale], dtype=torch.float32)

            # Apply pose normalization if enabled
            if self.pose_normalizer is not None:
                pose_8d = self.pose_normalizer.normalize(pose_8d, 'quaternion_translation_scale')
                pose_10d = self.pose_normalizer.normalize(pose_10d, '6d_translation_scale')
                pose_13d = self.pose_normalizer.normalize(pose_13d, '9d_translation_scale')

            processed_poses.append({
                "pose_0": pose_8d,
                "pose_10d": pose_10d,
                "pose_13d": pose_13d,
            })

        # Build output pack
        pack: Dict[str, Any] = {
            "x_0": (ss_latent - self.mean) / self.std if self.normalization is not None else ss_latent,
        }

        # Handle single-view vs multi-view output format
        if self.num_views == 1:
            # Single-view: no view dimension, matches legacy single-view API
            pack["cond"] = processed_conds[0]  # (3, H, W)
            pack["cond_mask"] = processed_masks[0]  # (1, H, W)
            pack["pose_0"] = processed_poses[0]["pose_0"]  # (8,)
            pack["pose_10d"] = processed_poses[0]["pose_10d"]  # (10,)
            pack["pose_13d"] = processed_poses[0]["pose_13d"]  # (13,)

            if all(pm is not None for pm in processed_pointmaps):
                pack["cond_pointmap"] = processed_pointmaps[0]  # (3, H, W)
                pack["cam2ncam"] = processed_cam2ncams[0]

            if all(ind is not None for ind in processed_indoors):
                pack["cond_indoors"] = processed_indoors[0]  # (3, H, W)

            # Also store pose_data as JSON string for collate_fn compatibility
            pack["pose_data"] = json.dumps(pose_data_list[selected_indices[0]])
            # Store view_metadata and aug_bbox for run_snapshot overlay rendering
            pack["view_metadata"] = processed_view_metadata[0]
            pack["aug_bbox"] = processed_aug_bboxes[0]
        else:
            # Multi-view: stack with view dimension
            pack["cond"] = torch.stack(processed_conds, dim=0)  # (num_views, 3, H, W)
            pack["cond_mask"] = torch.stack(processed_masks, dim=0)  # (num_views, 1, H, W)
            pack["num_views"] = self.num_views
            pack["pose_0"] = torch.stack([p["pose_0"] for p in processed_poses], dim=0)  # (num_views, 8)
            pack["pose_10d"] = torch.stack([p["pose_10d"] for p in processed_poses], dim=0)  # (num_views, 10)
            pack["pose_13d"] = torch.stack([p["pose_13d"] for p in processed_poses], dim=0)  # (num_views, 13)

            if all(pm is not None for pm in processed_pointmaps):
                pack["cond_pointmap"] = torch.stack(processed_pointmaps, dim=0)  # (num_views, 3, H, W)
                pack["cam2ncam"] = processed_cam2ncams

            if all(ind is not None for ind in processed_indoors):
                pack["cond_indoors"] = torch.stack(processed_indoors, dim=0)  # (num_views, 3, H, W)

            # Store view_metadata and aug_bbox for run_snapshot overlay rendering
            # For multi-view, use the first view's metadata (same object, different views)
            pack["view_metadata"] = processed_view_metadata[0]
            pack["aug_bbox"] = processed_aug_bboxes[0]

        # Depth type indicator
        pack["depth_type"] = "st" if use_stereo_depth else "normal"

        # Metadata
        if "sha256.txt" in sample:
            pack["sha256"] = sample["sha256.txt"].decode("utf-8")

        if "aesthetic_score.json" in sample:
            pack["aesthetic_score"] = float(sample["aesthetic_score.json"])

        if "captions.txt" in sample:
            pack["captions"] = sample["captions.txt"].decode("utf-8")

        if "dataset_type.txt" in sample:
            pack["dataset_type"] = sample["dataset_type.txt"].decode("utf-8")
        else:
            pack["dataset_type"] = "objectbased"

        # Filter by aesthetic score
        if (
            self.min_aesthetic_score
            and "aesthetic_score" in pack
            and pack["aesthetic_score"] < self.min_aesthetic_score
        ):
            return None

        return pack

    @staticmethod
    def _apply_color_jitter(img: torch.Tensor, params) -> torch.Tensor:
        """Apply pre-sampled ColorJitter params to a (3, H, W) tensor in [0, 1]."""
        import torchvision.transforms.functional as TF

        fn_idx, brightness, contrast, saturation, hue = params
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                img = TF.adjust_brightness(img, brightness)
            elif fn_id == 1 and contrast is not None:
                img = TF.adjust_contrast(img, contrast)
            elif fn_id == 2 and saturation is not None:
                img = TF.adjust_saturation(img, saturation)
            elif fn_id == 3 and hue is not None:
                img = TF.adjust_hue(img, hue)
        return img

    @staticmethod
    def _interleave_shard_urls(urls: list) -> list:
        """Interleave shard URLs so different datasets are evenly distributed.

        Without interleaving, brace expansion produces all shards from dataset A,
        then all from B, etc. With shardshuffle=100, the middle of a large dataset
        block (e.g. 15k Objaverse shards) never mixes with other datasets.

        This assigns each URL a proportionally-spaced position based on its dataset
        group, then sorts by position. The result is that every local window of
        shards contains a representative mix of all datasets.
        """
        from collections import defaultdict
        import os

        # Group by dataset directory (e.g. "ABO_wds", "HSSD_wds")
        groups = defaultdict(list)
        for url in urls:
            dataset_dir = os.path.basename(os.path.dirname(url))
            groups[dataset_dir].append(url)

        if len(groups) <= 1:
            return urls

        total = len(urls)
        indexed = []
        for group_idx, group_urls in enumerate(groups.values()):
            n = len(group_urls)
            stride = total / n
            # Offset by group_idx * small epsilon to break ties deterministically
            offset = group_idx * 0.01
            for i, url in enumerate(group_urls):
                indexed.append((i * stride + offset, url))

        indexed.sort(key=lambda x: x[0])
        return [url for _, url in indexed]

    def _filter_none(self, x):
        return x is not None

    def _build_dataset(self, shuffle: bool, shuffle_buffer: int, seed: int):
        import random

        # Pre-expand and interleave shard URLs for better multi-dataset mixing
        urls = self.shards
        if self.interleave_shards and isinstance(urls, str):
            from braceexpand import braceexpand
            urls = list(braceexpand(urls))
            urls = self._interleave_shard_urls(urls)

        ds = wds.WebDataset(
            urls,
            handler=_warn_and_count,
            nodesplitter=wds.shardlists.split_by_node,
            shardshuffle=self.shardshuffle,
            empty_check=False  # Allow workers with no shards (when #shards < #workers)
        )

        if shuffle:
            rng = random.Random(seed)
            # initial=min(...) so iteration starts quickly even if some
            # samples fail; the buffer fills gradually during training.
            ds = ds.shuffle(shuffle_buffer, initial=min(shuffle_buffer, 5), rng=rng)

        ds = ds.map(self._decode_sample).select(self._filter_none)
        return ds

    def __iter__(self):
        for sample in self.dataset:
            yield sample

    def __len__(self):
        if self.length_override is not None:
            return self.length_override
        raise TypeError("Length of WebDataset is unknown; provide length if needed.")

    @staticmethod
    def collate_fn(batch, split_size=None):
        """
        Collate function for multi-view batches.

        Input batch: list of dicts, each with:
            - cond: (num_views, 3, H, W)
            - cond_mask: (num_views, 1, H, W)
            - pose_0: (num_views, 8)
            - pose_10d: (num_views, 10)
            - pose_13d: (num_views, 13)
            - x_0: (8, 16, 16, 16)

        Output:
            - cond: (B, num_views, 3, H, W)
            - cond_mask: (B, num_views, 1, H, W)
            - pose_0: (B, num_views, 8)
            - pose_10d: (B, num_views, 10)
            - pose_13d: (B, num_views, 13)
            - x_0: (B, 8, 16, 16, 16)
        """
        if not batch:
            return {}

        common_keys = set(batch[0].keys())
        for sample in batch[1:]:
            common_keys &= set(sample.keys())

        pack: Dict[str, Any] = {}

        def stack_if_tensor(values):
            if isinstance(values[0], torch.Tensor):
                return torch.stack(values)
            return values

        for key in common_keys:
            values = [sample[key] for sample in batch]
            if any(v is None for v in values):
                continue
            pack[key] = stack_if_tensor(values)

        if split_size is None:
            return pack
        return [pack]

    def __str__(self):
        lines = [self.__class__.__name__]
        lines.append(f"  - Shards: {self.shards}")
        lines.append(f"  - Num views per sample: {self.num_views}")
        lines.append(f"  - Image size: {self.image_size}")
        lines.append(f"  - Pose variant: {self.pose_variant}")
        lines.append(f"  - mix_stereo_depth: {self.mix_stereo_depth}")
        lines.append(f"  - color_jitter: {self.color_jitter}")
        lines.append(f"  - Min aesthetic score: {self.min_aesthetic_score}")
        lines.append(f"  - Min visible fraction: {self.min_visible_fraction}")
        lines.append(f"  - Min visible pixels: {self.min_visible_pixels}")
        lines.append(f"  - Filter: {self.filter}")
        lines.append(f"  - Require pose data: {self.require_pose_data}")
        lines.append(f"  - Require pointmap: {self.require_pointmap}")
        if self.blacklisted_shas is not None:
            lines.append(f"  - Blacklist: enabled ({len(self.blacklisted_shas)} sha256s)")
        return "\n".join(lines)


class WebDatasetImageConditionedSparseStructureLatent(WebDatasetMultiViewSparseStructureLatent):
    """
    Single-view WebDataset loader (convenience wrapper).

    This is equivalent to WebDatasetMultiViewSparseStructureLatent with num_views=1.
    Outputs have no view dimension, matching the legacy single-view API:
        - cond: (3, H, W)
        - cond_mask: (1, H, W)
        - pose_0: (8,)
        - etc.

    Uses multi-view WebDataset format internally, randomly selecting 1 view at runtime.
    """

    def __init__(self, *args, num_views: int = 1, **kwargs):
        """
        Same as WebDatasetMultiViewSparseStructureLatent but defaults to num_views=1.

        Note: num_views is accepted for compatibility but always set to 1.
        """
        if num_views != 1:
            import warnings
            warnings.warn(
                f"WebDatasetImageConditionedSparseStructureLatent is for single-view training. "
                f"Use WebDatasetMultiViewSparseStructureLatent for num_views={num_views}."
            )
        super().__init__(*args, num_views=1, **kwargs)


class WebDatasetMultiViewStructuredLatent(WebDatasetMultiViewSparseStructureLatent):
    """
    WebDataset for Structured Latent (SLAT) training with multi-view support.

    SLAT latents are sparse (variable-length coords + feats per sample), stored as:
        - slat_coords.npy: (N, 3) int16 voxel coordinates
        - slat_feats.npy: (N, 8) float32 latent features

    The collate_fn creates a SparseTensor from batched coords+feats, matching
    the non-WDS SLat dataset's collate behavior.

    Use with --include-slat flag when running convert_all_datasets_to_wds.py.
    """

    def __init__(self, *args, max_num_voxels: int = 32768,
                 pretrained_slat_dec: str = 'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
                 **kwargs):
        if kwargs.get('color_jitter'):
            raise ValueError(
                "color_jitter is not supported for SLAT training: SLAT latents "
                "encode appearance, so jittering the conditioning image breaks "
                "input/target color consistency (SS-only augmentation)."
            )
        super().__init__(*args, **kwargs)
        self.max_num_voxels = max_num_voxels
        self.pretrained_slat_dec = pretrained_slat_dec
        self.slat_dec = None
        # Reshape normalization for sparse feats (N, 8) instead of dense (8, D, D, D)
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization["mean"]).reshape(1, -1)  # (1, 8)
            self.std = torch.tensor(self.normalization["std"]).reshape(1, -1)  # (1, 8)

    def _decode_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Decode a single WebDataset sample with sparse SLAT latent.

        Returns coords and feats instead of a dense x_0 tensor.
        """
        sample_key = sample.get("__key__", "<unknown>")

        # Check for required SLAT keys
        if "slat_coords.npy" not in sample or "slat_feats.npy" not in sample:
            return None

        # Load sparse SLAT latent
        try:
            slat_coords = torch.from_numpy(_load_npy(sample["slat_coords.npy"])).int()  # (N, 3)
            slat_feats = torch.from_numpy(_load_npy(sample["slat_feats.npy"])).float()  # (N, 8)
        except Exception as e:
            logger.warning("Skipping sample %s: failed to load SLAT latent: %s", sample_key, e)
            return None

        if slat_coords.ndim != 2 or slat_feats.ndim != 2:
            return None
        if slat_coords.shape[0] != slat_feats.shape[0]:
            return None
        if slat_coords.shape[0] > self.max_num_voxels:
            return None

        # Normalize feats
        if self.normalization is not None:
            slat_feats = (slat_feats - self.mean) / self.std

        # Inject a dummy ss_latent so parent's _decode_sample can process views.
        # Temporarily disable normalization since self.mean/std are reshaped for
        # sparse feats (1, 8) and would fail against the dense dummy (8, 16, 16, 16).
        dummy_ss = np.zeros((8, 16, 16, 16), dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, dummy_ss)
        sample["ss_latent.npy"] = buf.getvalue()

        saved_normalization = self.normalization
        self.normalization = None
        # Call parent for view processing (images, poses, pointmaps, etc.)
        result = super()._decode_sample(sample)
        self.normalization = saved_normalization
        if result is None:
            return None

        # Replace dense x_0 with sparse coords+feats
        del result["x_0"]
        result["coords"] = slat_coords
        result["feats"] = slat_feats

        return result

    @staticmethod
    def collate_fn(batch, split_size=None):
        """
        Collate function that creates SparseTensor from sparse coords+feats.

        Mirrors SLat.collate_fn behavior: variable-length coords+feats are
        concatenated with batch indices prepended to coords.
        """
        from recgen_inference.recgen_modules.modules.sparse import SparseTensor
        from ..utils.data_utils import load_balanced_group_indices

        if not batch:
            return {}

        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices(
                [b['coords'].shape[0] for b in batch], split_size
            )

        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}

            # Build SparseTensor from coords+feats
            coords_list = []
            feats_list = []
            layout = []
            start = 0
            for i, b in enumerate(sub_batch):
                coords_list.append(
                    torch.cat([
                        torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32),
                        b['coords']
                    ], dim=-1)
                )
                feats_list.append(b['feats'])
                layout.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]

            coords_cat = torch.cat(coords_list)
            feats_cat = torch.cat(feats_list)
            pack['x_0'] = SparseTensor(coords=coords_cat, feats=feats_cat)
            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            pack['x_0'].register_spatial_cache('layout', layout)

            # Stack other tensors normally, skip keys missing from any sample
            other_keys = [k for k in sub_batch[0].keys() if k not in ['coords', 'feats']]
            for k in other_keys:
                values = [b.get(k) for b in sub_batch]
                if any(v is None for v in values):
                    continue
                if isinstance(values[0], torch.Tensor):
                    pack[k] = torch.stack(values)
                elif isinstance(values[0], tuple):
                    pack[k] = tuple(
                        torch.stack([v[j] for v in values])
                        if isinstance(values[0][j], torch.Tensor)
                        else [v[j] for v in values]
                        for j in range(len(values[0]))
                    )
                else:
                    pack[k] = values

            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs

    def _loading_slat_dec(self):
        if self.slat_dec is not None:
            return
        from recgen_inference.recgen_modules import models
        decoder = models.from_pretrained(self.pretrained_slat_dec)
        self.slat_dec = decoder.cuda().eval()

    def _delete_slat_dec(self):
        del self.slat_dec
        self.slat_dec = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        """Decode SparseTensor latents to Gaussian representations via SLAT decoder."""
        from recgen_inference.recgen_modules.modules.sparse.basic import SparseTensor
        self._loading_slat_dec()
        reps = []
        if self.normalization is not None:
            z = z.replace(z.feats * self.std.to(z.device) + self.mean.to(z.device))
        for i in range(0, z.shape[0], batch_size):
            reps.append(self.slat_dec(z[i:i+batch_size]))
        reps = sum(reps, [])
        self._delete_slat_dec()
        return reps

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        """
        Visualize sparse SLAT samples by decoding through the SLAT Gaussian
        decoder and rendering with full appearance (color/texture).

        Requires optional visualization dependencies (utils3d + a Gaussian
        rasterizer). Raises ImportError when they are missing; snapshot calls
        in the trainer are exception-guarded, so training is unaffected.
        """
        from recgen_inference.recgen_modules.utils.render_utils import get_renderer
        import utils3d

        x_0 = x_0 if not isinstance(x_0, dict) else x_0['x_0']
        reps = self.decode_latent(x_0.cuda())

        # Build cameras
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, p in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(p),
                np.cos(yaw) * np.cos(p),
                np.sin(p),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(40)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        renderer = get_renderer(reps[0])
        images = []
        for representation in reps:
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)

        return torch.stack(images)

    def __str__(self):
        lines = super().__str__().split("\n")
        lines[0] = self.__class__.__name__ + " (SLAT, sparse)"
        lines.append(f"  - Max num voxels: {self.max_num_voxels}")
        return "\n".join(lines)
