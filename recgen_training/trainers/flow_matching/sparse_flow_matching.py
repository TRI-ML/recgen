from typing import *
import os
import copy
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from recgen_inference.recgen_modules.modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin, MultiImageConditionedMixin
from ...utils import dist_utils


class SparseFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective.
    
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
        use_pose_conditioning (bool): Whether to use pose as conditioning. Default: False.
        pose_representation (str): Which pose representation to use: '9d_translation_scale' (rotation matrix),
                                   '6d_translation_scale' (6D rotation), or 'quaternion_translation_scale' (quaternion). 
                                     Default: 'quaternion_translation_scale'.
    """
    
    def __init__(self, *args, use_pose_conditioning: bool = False, pose_representation: str = 'quaternion_translation_scale', use_masked_rgb: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_pose_conditioning = use_pose_conditioning
        self.pose_representation = pose_representation
        self.use_masked_rgb = use_masked_rgb
        
        # Validate pose_representation
        valid_keys = ['quaternion_translation_scale', '6d_translation_scale', '9d_translation_scale']
        if self.pose_representation not in valid_keys:
            raise ValueError(f"pose_representation must be one of {valid_keys}, got {self.pose_representation}")
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        from torch.utils.data import IterableDataset
        is_iterable = isinstance(self.dataset, IterableDataset)
        
        # For IterableDataset (e.g., WebDataset), sampler must be None
        if is_iterable:
            self.data_sampler = None
            num_workers = min(8, max(1, int(os.cpu_count() / max(torch.cuda.device_count(), 1)) // 2))
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size_per_gpu,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=num_workers > 0,
                collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
                sampler=None,
                prefetch_factor=2 if num_workers > 0 else None,
            )
        else:
            self.data_sampler = BalancedResumableSampler(
                self.dataset,
                shuffle=True,
                batch_size=self.batch_size_per_gpu,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size_per_gpu,
                num_workers=min(int(np.ceil(os.cpu_count() / torch.cuda.device_count())), 12),
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
                collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
                sampler=self.data_sampler,
            )
        self.data_iterator = cycle(self.dataloader)
        
    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        cond_pointmap=None,
        cond_mask=None,
        pose_0=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            cond_pointmap: The [N x 3 x H x W] tensor of pointmap conditioning.
            cond_mask: The [N x 1 x H x W] tensor of mask conditioning.
            pose_0: The [N x 8] tensor of pose conditioning (quaternion, translation, scale).
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)

        # Apply mask to image before encoding (masked RGB: zero out background)
        if self.use_masked_rgb and cond_mask is not None:
            cond = cond * (cond_mask > 0.5).float()
            cond_mask = None

        cond = self.get_cond(cond, **kwargs)

        # Add pointmap conditioning if present
        # Note: cond already has CLS+register tokens stripped by encode_image,
        # so cond and cond_pointmap have the same number of patch tokens.
        if cond_pointmap is not None:
            cond_pointmap = self.get_pointmap_cond(cond_pointmap, **kwargs)
            cond += cond_pointmap

        # Add mask conditioning if present
        if cond_mask is not None:
            cond_mask = self.get_mask_cond(cond_mask, **kwargs)
            cond += cond_mask

        #Note: pose concat  should be done after we add pointmap and mask conditioning

        # Add pose conditioning if enabled and present (concatenate as additional token).
        # Skipped when pose_0 is None (e.g. multi-view trainer uses additive pose via get_cond instead).
        if self.use_pose_conditioning and pose_0 is not None:
            pose_13d = kwargs.pop('pose_13d', None)
            pose_10d = kwargs.pop('pose_10d', None)
            pose_8d = pose_0

            # Select pose based on configuration
            if self.pose_representation == '9d_translation_scale':
                assert pose_13d is not None, "pose_13d is required for 9d_translation_scale"
                cond_pose = self.get_pose_cond(pose_13d, **kwargs)
                cond = torch.cat([cond, cond_pose], dim=1)
            elif self.pose_representation == '6d_translation_scale':
                assert pose_10d is not None, "pose_10d is required for 6d_translation_scale"
                cond_pose = self.get_pose_cond(pose_10d, **kwargs)
                cond = torch.cat([cond, cond_pose], dim=1)
            elif self.pose_representation == 'quaternion_translation_scale':
                assert pose_8d is not None, "pose_8d is required for quaternion_translation_scale"
                cond_pose = self.get_pose_cond(pose_8d, **kwargs)
                cond = torch.cat([cond, cond_pose], dim=1)
            else:
                raise ValueError(f"Invalid pose representation: {self.pose_representation}")
        
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        # log loss with time bins (under _additional to keep loss/ panel clean)
        additional = {}
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                additional[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        if additional:
            terms["_additional"] = additional

        return terms, {}

    def _prepare_snapshot_data(self, data: dict) -> None:
        """Hook for subclasses to pre-process snapshot batch data before get_inference_cond."""
        if self.use_masked_rgb and 'cond_mask' in data and 'cond' in data:
            data['cond'] = data['cond'] * (data['cond_mask'] > 0.5).float()
            del data['cond_mask']

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        from torch.utils.data import IterableDataset
        is_iterable = isinstance(self.dataset, IterableDataset)

        # Rebuild dataset pipeline with fixed seed for deterministic snapshots.
        # shuffle=False avoids filling a 200-sample shuffle buffer over S3
        # before yielding the first sample; the fixed seed=0 already ensures
        # deterministic shard ordering via shardshuffle.
        snapshot_dataset = copy.deepcopy(self.dataset)
        if is_iterable and hasattr(snapshot_dataset, '_build_dataset'):
            snapshot_dataset.dataset = snapshot_dataset._build_dataset(
                shuffle=False, shuffle_buffer=0, seed=0,
            )

        dataloader = DataLoader(
            snapshot_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        sample = []
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
                elif isinstance(v, list):
                    processed_data[k] = v[:batch]
                else:
                    processed_data[k] = v[:batch].cuda()
            data = processed_data
            noise = data['x_0'].replace(torch.randn_like(data['x_0'].feats))
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            self._prepare_snapshot_data(data)
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
            
            # Add pose conditioning if enabled and present (concat path).
            # Skipped when pose keys were already consumed by _prepare_snapshot_data (additive path).
            if self.use_pose_conditioning:
                pose_13d = data.pop('pose_13d', None)
                pose_10d = data.pop('pose_10d', None)
                pose_8d = data.pop('pose_0', None)
                if self.pose_representation == '9d_translation_scale' and pose_13d is not None:
                    cond_pose = self.get_pose_cond(pose_13d)
                    args['cond'] = torch.cat([args['cond'], cond_pose], dim=1)
                elif self.pose_representation == '6d_translation_scale' and pose_10d is not None:
                    cond_pose = self.get_pose_cond(pose_10d)
                    args['cond'] = torch.cat([args['cond'], cond_pose], dim=1)
                elif self.pose_representation == 'quaternion_translation_scale' and pose_8d is not None:
                    cond_pose = self.get_pose_cond(pose_8d)
                    args['cond'] = torch.cat([args['cond'], cond_pose], dim=1)
            
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=50, cfg_strength=3.0, verbose=verbose,
            )
            sample.append(res.samples)

        sample_gt = sp.sparse_cat(sample_gt)
        sample = sp.sparse_cat(sample)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))

        if snapshot_sha256s:
            print(f"Snapshot sha256s ({len(snapshot_sha256s)} samples):")
            for idx, sha in enumerate(snapshot_sha256s):
                print(f"  [{idx}] {sha}")

        return sample_dict


class SparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective and classifier-free guidance.
    
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


class ImageConditionedSparseFlowMatchingCFGTrainer(ImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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


class MultiImageConditionedSparseFlowMatchingCFGTrainer(MultiImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse multi-image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    Conditions on N images based on configurable image_keys.

    Supports per-view additive pose conditioning when the denoiser has use_pose_embedder=True.
    Pose embeddings are added to each view's DINO tokens (like frame_token_embedder).
    """

    def training_losses(self, x_0, pose_0=None, cond=None, cond_pointmap=None, cond_mask=None, **kwargs):
        B = x_0.shape[0]

        # Shared CFG mask: all modalities drop together when True
        shared_cfg_mask = np.random.rand(B) < self.p_uncond
        kwargs['_shared_cfg_mask'] = shared_cfg_mask

        # Per-sample view dropout (independent of CFG)
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

        # Prepare per-view pose for additive conditioning via get_cond
        raw_denoiser = dist_utils.unwrap_dist(self.training_models['denoiser'])
        if getattr(raw_denoiser, 'use_pose_embedder', False):
            # Select the right pose representation
            if self.pose_representation == '9d_translation_scale':
                pose = kwargs.pop('pose_13d', None)
            elif self.pose_representation == '6d_translation_scale':
                pose = kwargs.pop('pose_10d', None)
            else:
                pose = pose_0
            # Clean up unused pose keys from kwargs
            kwargs.pop('pose_13d', None)
            kwargs.pop('pose_10d', None)

            if pose is not None:
                kwargs['_pose_for_cond'] = pose  # picked up by get_cond
            # Pass pose_0=None to skip the base trainer's concatenation path
            pose_0 = None

        return super().training_losses(x_0, pose_0=pose_0, cond=cond,
                                       cond_pointmap=cond_pointmap, cond_mask=cond_mask, **kwargs)

    def _prepare_snapshot_data(self, data: dict) -> None:
        """Pre-process snapshot data: move pose into _pose_for_cond for additive conditioning."""
        super()._prepare_snapshot_data(data)
        raw_denoiser = dist_utils.unwrap_dist(self.models['denoiser'])
        if not getattr(raw_denoiser, 'use_pose_embedder', False):
            return

        # Select the right pose representation
        if self.pose_representation == '9d_translation_scale':
            pose = data.pop('pose_13d', None)
        elif self.pose_representation == '6d_translation_scale':
            pose = data.pop('pose_10d', None)
        else:
            pose = data.pop('pose_0', None)
        # Clean up remaining pose keys so the base run_snapshot doesn't concatenate them
        data.pop('pose_13d', None)
        data.pop('pose_10d', None)
        data.pop('pose_0', None)

        if pose is not None:
            data['_pose_for_cond'] = pose
