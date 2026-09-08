from typing import *
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from recgen_inference.recgen_modules.pipelines import samplers
from ...utils.general_utils import dict_reduce
from ...utils import dist_utils
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin, MultiImageConditionedMixin
from recgen_inference.recgen_modules.utils.pose_utils import (
    compute_geodesic_rotation_errors_batch, get_pose_dimensions
)


def compute_pose_statistics(pose_0: torch.Tensor, dataset_types: Optional[List[str]] = None, pose_representation: str = 'quaternion_translation_scale') -> Dict[str, torch.Tensor]:
    """
    Compute pose statistics for logging, with optional breakdown by dataset type.

    Args:
        pose_0: Tensor of shape [B, D] containing pose data where D depends on representation:
                - 8D (quaternion_translation_scale): [quat(4), trans(3), scale(1)]
                - 10D (6d_translation_scale): [6d_rot(6), trans(3), scale(1)]
                - 13D (9d_translation_scale): [rot_9d(9), trans(3), scale(1)]
        dataset_types: Optional list of dataset type strings (e.g., ["objectbased", "partbased"])
                       for per-type statistics breakdown.
        pose_representation: Pose format string

    Returns:
        Dictionary of pose statistics tensors:
        - pose_rot_norm: Mean rotation representation norm
        - pose_trans_norm: Mean translation vector magnitude
        - pose_scale: Mean scale value
        - pose_rot_norm_{dtype}: Per-dataset-type rotation norm (if dataset_types provided)
        - pose_trans_norm_{dtype}: Per-dataset-type translation norm (if dataset_types provided)
        - pose_scale_{dtype}: Per-dataset-type scale (if dataset_types provided)
    """
    stats = {}

    # Get pose dimensions
    rot_dim, trans_start, scale_idx, _ = get_pose_dimensions(pose_representation)

    # Compute overall statistics
    rot_norm = pose_0[:, :rot_dim].norm(dim=1)  # rotation representation norm
    trans_norm = pose_0[:, trans_start:trans_start+3].norm(dim=1)  # translation norm
    scale = pose_0[:, scale_idx]  # scale

    stats["pose_rot_norm"] = rot_norm.mean()
    stats["pose_trans_norm"] = trans_norm.mean()
    stats["pose_scale"] = scale.mean()

    # Break down by dataset type if provided
    if dataset_types is not None:
        for dtype in ["objectbased", "partbased"]:
            mask = torch.tensor([dt == dtype for dt in dataset_types], device=pose_0.device)
            if mask.any():
                stats[f"pose_rot_norm_{dtype}"] = rot_norm[mask].mean()
                stats[f"pose_trans_norm_{dtype}"] = trans_norm[mask].mean()
                stats[f"pose_scale_{dtype}"] = scale[mask].mean()

    return stats


class FlowMatchingTrainer(BasicTrainer):
    """
    Trainer for diffusion model with flow matching objective.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        pose_alpha (float): Weight for pose prediction loss.
    """
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
        pose_alpha: float = 0.01,
        pose_representation: str = 'quaternion_translation_scale',
        use_pose_normalization: bool = False,
        pose_normalization_config: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min
        self.pose_alpha = pose_alpha
        self.pose_representation = pose_representation
        self.use_pose_normalization = use_pose_normalization
        self.pose_normalization_config = pose_normalization_config

        # Validate pose_conditioning_key
        valid_keys = ['quaternion_translation_scale', '6d_translation_scale', '9d_translation_scale']
        if self.pose_representation not in valid_keys:
            raise ValueError(f"pose_representation must be one of {valid_keys}, got {self.pose_representation}")

        # Initialize pose normalizer if enabled
        self.pose_normalizer = None
        self._pose_normalizer_initialized = False

        if self.use_pose_normalization:
            # Priority: 1. Explicit config file, 2. Dataset's cached normalizer
            if self.pose_normalization_config is not None:
                # Explicit config file provided - use it
                from recgen_inference.recgen_modules.utils.pose_utils import load_pose_normalization_stats, PoseNormalizer
                stats = load_pose_normalization_stats(self.pose_normalization_config)
                self.pose_normalizer = PoseNormalizer(stats)
                self._pose_normalizer_initialized = True
            elif hasattr(self.dataset, 'pose_normalizer') and self.dataset.pose_normalizer is not None:
                # Use dataset's cached pose normalizer (computed/loaded automatically)
                self.pose_normalizer = self.dataset.pose_normalizer
                self._pose_normalizer_initialized = True
                if self.is_master:
                    print("Using pose normalizer from dataset (auto-cached stats)")
            else:
                raise ValueError(
                    "use_pose_normalization=True but no normalizer available. "
                    "Either provide pose_normalization_config or use a dataset with pose_stats caching."
                )

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        return cond

    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        return {'cond': cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerSampler(self.sigma_min)

    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps.
        """
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def training_losses(
        self,
        x_0: torch.Tensor,
        pose_0: torch.Tensor = None,
        cond=None,
        cond_pointmap=None,
        cond_mask=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)

        # rewrite pose_0 based on representation
        if self.pose_representation == '9d_translation_scale':
            pose_0 = kwargs.pop('pose_13d', None)
        elif self.pose_representation == '6d_translation_scale':
            pose_0 = kwargs.pop('pose_10d', None)
        noise_pose = torch.randn_like(pose_0) if pose_0 is not None else None

        pose_t = self.diffuse(pose_0, t, noise=noise_pose) if pose_0 is not None else None

        cond = self.get_cond(cond, **kwargs)
        # Add pointmap conditioning if present
        if cond_pointmap is not None:
            cond_pointmap = self.get_pointmap_cond(cond_pointmap, **kwargs)
            cond += cond_pointmap

        # Add mask conditioning only if the model has a mask embedder
        raw_denoiser = dist_utils.unwrap_dist(self.training_models['denoiser'])
        if getattr(raw_denoiser, 'use_mask_embedder', True):
            if cond_mask is None:
                raise RuntimeError("cond_mask is None but model has mask_embedder! All samples should have masks.")
            cond_mask = self.get_mask_cond(cond_mask, **kwargs)
            cond += cond_mask

        pred, pred_pose = self.training_models['denoiser'](x_t, pose_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape, f"pred.shape: {pred.shape}, noise.shape: {noise.shape}, x_0.shape: {x_0.shape}"
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"].clone()

        if pose_0 is not None:
            assert pred_pose.shape == pose_0.shape, \
                f"pred_pose.shape: {pred_pose.shape}, pose_0.shape: {pose_0.shape}"
            target_pose = self.get_v(pose_0, noise_pose, t)

            # Apply per-token pose loss mask if provided (e.g. dropped second view)
            pose_loss_mask = kwargs.get('_pose_loss_mask', None)
            if pose_loss_mask is not None and pose_0.ndim == 3:
                # pose_loss_mask: (B, K), pose errors: (B, K, D)
                per_elem = (pred_pose - target_pose) ** 2
                mask = pose_loss_mask.unsqueeze(-1).to(per_elem.device)  # (B, K, 1)
                terms["mse_pose"] = (per_elem * mask).sum() / mask.sum().clamp(min=1) / pose_0.shape[-1]
            else:
                terms["mse_pose"] = F.mse_loss(pred_pose, target_pose)
            terms["loss"] += self.pose_alpha * terms["mse_pose"]

        # Additional metrics (logged under loss_additional/ to keep loss/ panel clean)
        additional = {}

        if pose_0 is not None:
            # Log pose statistics by dataset type
            # For multi-pose tokens, use first token only to keep stats aligned with dataset_type
            pose_for_stats = pose_0[:, 0] if pose_0.ndim == 3 else pose_0
            pose_stats = compute_pose_statistics(pose_for_stats, kwargs.get('dataset_type', None), self.pose_representation)
            additional.update(pose_stats)

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                additional[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        if additional:
            terms["_additional"] = additional

        return terms, {}

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        from torch.utils.data import IterableDataset
        is_iterable = isinstance(self.dataset, IterableDataset)

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False if is_iterable else True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        pose_samples = []
        pose_gt = []
        cond_vis = []
        snapshot_sha256s = []
        data_iter = iter(dataloader)
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(data_iter)
            # Log sha256s for debugging
            if 'sha256' in data:
                snapshot_sha256s.extend(data['sha256'][:batch])
            # Handle different data types: tensors, tuples (dual image), and lists
            processed_data = {}
            for k, v in data.items():
                if isinstance(v, tuple):
                    # Dual image conditioning: (full_image, part_image)
                    processed_data[k] = tuple(t[:batch].cuda() for t in v)
                elif isinstance(v, torch.Tensor):
                    processed_data[k] = v[:batch].cuda()
                else:
                    processed_data[k] = v[:batch]
            data = processed_data
            noise = torch.randn_like(data['x_0'])

            # Generate pose noise if model supports it
            pose_noise = None
            if hasattr(self.models['denoiser'], 'pose_channels'):
                num_pose_tokens = getattr(self.models['denoiser'], 'num_pose_tokens', 1)
                if num_pose_tokens > 1:
                    pose_noise = torch.randn(
                        data['x_0'].shape[0],
                        num_pose_tokens,
                        self.models['denoiser'].pose_channels,
                        device=data['x_0'].device,
                        dtype=data['x_0'].dtype
                    )
                else:
                    pose_noise = torch.randn(
                        data['x_0'].shape[0],
                        self.models['denoiser'].pose_channels,
                        device=data['x_0'].device,
                        dtype=data['x_0'].dtype
                    )

            sample_gt.append(data['x_0'])

            # Select the correct precomputed pose representation from dataset
            # For multi-view data, poses have shape (B, num_views, D) - use first view's pose
            if self.pose_representation == 'quaternion_translation_scale':
                pose = data['pose_0']  # 8D: [quat(4), trans(3), scale(1)]
            elif self.pose_representation == '6d_translation_scale':
                pose = data['pose_10d']  # 10D: [6d_rot(6), trans(3), scale(1)]
            elif self.pose_representation == '9d_translation_scale':
                pose = data['pose_13d']  # 13D: [9d_rot(9), trans(3), scale(1)]
            else:
                raise ValueError(f"Unknown pose_representation: {self.pose_representation}")
            # Handle multi-view: use first view's pose for GT metrics/rendering
            if pose.ndim == 3:  # (B, num_views, D)
                pose = pose[:, 0]  # (B, D) - first view for metrics
            pose_gt.append(pose)

            cond_vis.append(self.vis_cond(**data))

            del data['x_0']
            # Remove pose_0 from data as it's ground truth, not for inference
            if 'pose_0' in data:
                del data['pose_0']

            args = self.get_inference_cond(**data)

            # Merge pointmap conditioning into cond if present
            cond_pointmap = data.pop('cond_pointmap', None)
            if cond_pointmap is not None:
                no_drop = np.zeros(cond_pointmap.shape[0], dtype=bool)
                cond_pointmap = self.get_pointmap_cond(cond_pointmap, _shared_cfg_mask=no_drop, _cfg_mask=no_drop)
                args['cond'] += cond_pointmap

            # Add mask conditioning if present
            cond_mask = data.pop('cond_mask', None)
            if cond_mask is not None:
                cond_mask = self.get_mask_cond(cond_mask)
                args['cond'] += cond_mask

            # Add pose noise to inference args if generated
            if pose_noise is not None:
                args['pose'] = pose_noise

            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=50, cfg_strength=3.0, verbose=verbose,
            )
            sample.append(res.samples)
            pose_samples.append(res.final_pose)
        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)

        # Concatenate pose samples and apply denormalization if enabled
        pose_samples = torch.cat(pose_samples, dim=0) if pose_samples else None
        # For multi-pose tokens, use first pose token for metrics/rendering
        if pose_samples is not None and pose_samples.ndim == 3:
            pose_samples = pose_samples[:, 0]  # (N, D)
        if pose_samples is not None and self.pose_normalizer is not None:
            pose_samples = self.pose_normalizer.denormalize(pose_samples, self.pose_representation)

        # Concatenate and potentially denormalize ground truth poses
        pose_gt = torch.cat(pose_gt, dim=0)
        if self.pose_normalizer is not None:
            pose_gt = self.pose_normalizer.denormalize(pose_gt, self.pose_representation)

        # Compute pose error metrics (works for all representations)

        # Get pose dimensions
        rot_dim, trans_start, scale_idx, _ = get_pose_dimensions(self.pose_representation)

        # Set rotation metric name based on representation
        rot_metric_name = {
            'quaternion_translation_scale': 'quat_l2',
            '6d_translation_scale': 'rot_6d_l2',
            '9d_translation_scale': 'rot_9d_l2'
        }[self.pose_representation]

        # Compute rotation error (L2 in representation space)
        rot_l2 = torch.norm(pose_samples[:, :rot_dim] - pose_gt[:, :rot_dim], dim=1).mean()

        # Compute translation and scale errors
        trans_l2 = torch.norm(pose_samples[:, trans_start:trans_start+3] - pose_gt[:, trans_start:trans_start+3], dim=1).mean()
        scale_l2 = torch.abs(pose_samples[:, scale_idx] - pose_gt[:, scale_idx]).mean()

        # Overall pose error
        pose_error = torch.norm(pose_samples - pose_gt, dim=1).mean()

        # Compute geodesic rotation error (rotation-matrix-based, representation-agnostic)
        rot_geodesic_errors = compute_geodesic_rotation_errors_batch(
            pose_samples, pose_gt, self.pose_representation, return_degrees=True
        )
        rot_geodesic = torch.tensor(np.nanmean(rot_geodesic_errors)) if not np.all(np.isnan(rot_geodesic_errors)) else torch.tensor(float('nan'))

        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
            'pose_samples': {'value': pose_samples, 'type': 'pose'},
            'pose_gt': {'value': pose_gt, 'type': 'pose'},
            # Pose metrics for wandb (representation-agnostic)
            'pose_error_total': {'value': pose_error.item(), 'type': 'metric'},
            f'pose_error_{rot_metric_name}': {'value': rot_l2.item(), 'type': 'metric'},
            'pose_error_rotation_geodesic_deg': {'value': rot_geodesic.item() if not torch.isnan(rot_geodesic) else 0.0, 'type': 'metric'},
            'pose_error_translation_l2': {'value': trans_l2.item(), 'type': 'metric'},
            'pose_error_scale_l2': {'value': scale_l2.item(), 'type': 'metric'},
        }

        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))

        if snapshot_sha256s:
            rank = getattr(self, 'rank', 0)
            world_size = getattr(self, 'world_size', 1)
            print(f"Snapshot sha256s (rank {rank}/{world_size}, {len(snapshot_sha256s)} samples):")
            for idx, sha in enumerate(snapshot_sha256s):
                global_idx = rank * len(snapshot_sha256s) + idx
                print(f"  [{global_idx}] {sha}")

        return sample_dict


class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass


class ImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with flow matching objective and classifier-free guidance.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass


class MultiImageConditionedFlowMatchingCFGTrainer(MultiImageConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for multi-image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    Conditions on N images based on configurable image_keys.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """

    def training_losses(self, x_0, pose_0=None, cond=None, cond_pointmap=None, cond_mask=None, **kwargs):
        """
        Override to handle multi-view inputs where poses have shape (B, num_views, D).
        Uses the first view's pose as the target for denoising.

        Two independent per-sample dropout mechanisms:
        1. Shared CFG (p_uncond): drops ALL modalities → unconditional
        2. Second view drop (p_single_view): independently drops one view
        """
        B = x_0.shape[0]

        # Shared CFG mask: all modalities drop together when True
        shared_cfg_mask = np.random.rand(B) < self.p_uncond
        kwargs['_shared_cfg_mask'] = shared_cfg_mask

        # Per-sample second view dropout (independent of CFG, like any other modality)
        drop_view_per_sample = torch.full((B,), -2, dtype=torch.long)  # -2 = keep all views
        if cond is not None and isinstance(cond, torch.Tensor) and cond.ndim == 5:
            num_views = cond.shape[1]
            if num_views > 1:
                drop_second = torch.rand(B) < self.p_single_view
                if self.single_view_drop_mode == "last":
                    drop_view_per_sample[drop_second] = num_views - 1
                else:  # "random"
                    drop_view_per_sample[drop_second] = torch.randint(0, num_views, (drop_second.sum(),))
        kwargs['_drop_view_per_sample'] = drop_view_per_sample

        # Handle multi-view pose tensors
        # When model has num_pose_tokens > 1, keep all views; otherwise use first view only
        raw_denoiser = dist_utils.unwrap_dist(self.training_models['denoiser'])
        num_pose_tokens = getattr(raw_denoiser, 'num_pose_tokens', 1)

        if pose_0 is not None and pose_0.ndim == 3:  # (B, num_views, D)
            if num_pose_tokens == 1:
                pose_0 = pose_0[:, 0]  # Use first view's pose (B, D)
            else:
                pose_0 = pose_0[:, :num_pose_tokens]  # (B, num_pose_tokens, D)

        for pose_key in ['pose_10d', 'pose_13d']:
            if pose_key in kwargs and kwargs[pose_key] is not None:
                pose = kwargs[pose_key]
                if pose.ndim == 3:  # (B, num_views, D)
                    if num_pose_tokens == 1:
                        kwargs[pose_key] = pose[:, 0]
                    else:
                        kwargs[pose_key] = pose[:, :num_pose_tokens]

        # Build per-token pose loss mask for multi-pose: skip loss for dropped views
        if num_pose_tokens > 1:
            # Start with all ones: (B, num_pose_tokens)
            pose_loss_mask = torch.ones(B, num_pose_tokens, dtype=torch.float32)
            # Zero out tokens whose corresponding view was dropped
            for k in range(num_pose_tokens):
                pose_loss_mask[:, k][drop_view_per_sample == k] = 0.0
            # NOTE: Do NOT mask CFG-dropped samples. The model needs to learn
            # unconditional pose prediction for CFG to work at inference time.
            # (Volume loss is always computed for CFG samples — pose should be too.)
            kwargs['_pose_loss_mask'] = pose_loss_mask

        return super().training_losses(x_0, pose_0, cond, cond_pointmap, cond_mask, **kwargs)
