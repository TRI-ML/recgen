from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from einops import rearrange
from ....utils import dist_utils


class ImageConditionedMixin:
    """
    Mixin for image-conditioned models.
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(self, *args, image_cond_model: str = 'dinov2_vitl14_reg', **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        
    @staticmethod
    def prepare_for_training(image_cond_model: str, **kwargs):
        """
        Prepare for training.
        """
        if hasattr(super(ImageConditionedMixin, ImageConditionedMixin), 'prepare_for_training'):
            super(ImageConditionedMixin, ImageConditionedMixin).prepare_for_training(**kwargs)
        # download the model
        torch.hub.load('facebookresearch/dinov2', image_cond_model, pretrained=True)
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='xFormers is available')
            with dist_utils.local_master_first():
                dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image using DINOv2.

        Returns only patch tokens (no CLS or register tokens).
        DINOv2 outputs (B, 1+4+num_patches, dim) = (B, 5+seq_len, dim).
        We return (B, seq_len, dim) by stripping the first 5 tokens.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list) and len(image) > 0:
            if isinstance(image[0], Image.Image):
                # List of PIL images
                image = [i.resize((518, 518), Image.LANCZOS) for i in image]
                image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
                image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
                image = torch.stack(image).cuda()
            elif isinstance(image[0], torch.Tensor):
                # List of tensors - concatenate along batch dimension
                # E.g., [full_imgs(B,C,H,W), part_imgs(B,C,H,W)] -> (2B,C,H,W)
                image = torch.cat(image, dim=0).cuda()
            else:
                raise ValueError(f"Unsupported type in image list: {type(image[0])}")
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        if self.image_cond_model is None:
            self._init_image_cond_model()
        image = self.image_cond_model['transform'](image).cuda()
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        # Return only patch tokens, strip CLS (1) + registers (4)
        return patchtokens[:, 5:]
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_pointmap_cond(self, pointmap, **kwargs):
        """
        Get the pointmap conditioning data.
        """
        denoiser = self.training_models.get('denoiser') or self.models.get('denoiser')
        #TODO: is it correct version for DDP?
        cond = dist_utils.unwrap_dist(denoiser).encode_pointmap(pointmap)
        # Add random dropping for CFG
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        
        return cond

    def get_mask_cond(self, mask, **kwargs):
        """
        Get the mask conditioning data.
        Supports _cfg_mask for coordinated dropout (mask follows image).
        """
        denoiser = self.training_models.get('denoiser') or self.models.get('denoiser')
        cond = dist_utils.unwrap_dist(denoiser).encode_mask(mask)

        cfg_mask = kwargs.pop('_cfg_mask', None)
        if cfg_mask is not None:
            mask_tensor = torch.tensor(list(cfg_mask), device=cond.device, dtype=torch.bool)
            mask_tensor = mask_tensor.reshape(-1, *[1] * (cond.ndim - 1))
            cond = torch.where(mask_tensor, torch.zeros_like(cond), cond)

        return cond

    def get_pose_cond(self, pose, **kwargs):
        """
        Get the pose conditioning data.

        Args:
            pose: Pose tensor of shape (B, 8) containing [quaternion(4), translation(3), scale(1)]

        Returns:
            Encoded pose features of shape (B, 1, C)
        """
        denoiser = self.training_models.get('denoiser') or self.models.get('denoiser')
        cond = dist_utils.unwrap_dist(denoiser).encode_pose(pose)
        
        return cond

    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {'image': {'value': cond, 'type': 'image'}}


class MultiImageConditionedMixin(ImageConditionedMixin):
    """Multi-image conditioning: encodes N images and concatenates features"""

    def __init__(self, *args, p_single_view: float = 0.0, p_drop_all_views: float = 0.0,
                 single_view_drop_mode: str = "last", **kwargs):
        """
        Args:
            p_single_view: Probability of dropping to single view during training (0.0 = always multi-view)
            p_drop_all_views: Probability of dropping all views (unconditional, all zeros)
            single_view_drop_mode: Which view to drop - "last" (always drop last view) or "random"

        Example for 60/30/10 split:
            p_single_view=0.3, p_drop_all_views=0.1 → 60% both views, 30% single, 10% uncond
        """
        super().__init__(*args, **kwargs)
        self.p_single_view = p_single_view
        self.p_drop_all_views = p_drop_all_views
        self.single_view_drop_mode = single_view_drop_mode

    def get_cond(self, cond, **kwargs):
        """
        Get image conditioning with shared CFG dropout and per-sample view dropout.
        """
        num_views = cond.shape[1] if isinstance(cond, torch.Tensor) and cond.ndim == 5 else 1

        encoded = self.encode_image(cond)

        if num_views > 1:
            denoiser = self.training_models.get('denoiser') or self.models.get('denoiser')
            raw_denoiser = dist_utils.unwrap_dist(denoiser)

            # Apply per-frame token-type embeddings outside no_grad so gradients flow
            if getattr(raw_denoiser, 'use_frame_token_embedder', False):
                S = encoded.shape[1] // num_views
                stacked = encoded.reshape(encoded.shape[0], num_views, S, encoded.shape[2])
                encoded = raw_denoiser.encode_frame_tokens(stacked)

            # Apply per-view pose embeddings (additive, like frame tokens)
            pose = kwargs.pop('_pose_for_cond', None)
            if pose is not None and getattr(raw_denoiser, 'use_pose_embedder', False):
                S = encoded.shape[1] // num_views
                stacked = encoded.reshape(encoded.shape[0], num_views, S, encoded.shape[2])
                pose_emb = raw_denoiser.encode_pose(pose)  # (B, K, C)
                stacked = stacked + pose_emb[:, :, None, :]  # broadcast over N patches
                encoded = stacked.reshape(encoded.shape[0], num_views * S, encoded.shape[2])

        kwargs['neg_cond'] = torch.zeros_like(encoded)

        # Route shared CFG mask (all modalities drop together)
        shared_mask = kwargs.get('_shared_cfg_mask')
        if shared_mask is not None:
            kwargs['_cfg_mask'] = shared_mask  # CFG will pop this

        # Call grandparent's get_cond (skip ImageConditionedMixin since we already encoded)
        encoded = super(ImageConditionedMixin, self).get_cond(encoded, **kwargs)

        # Apply per-sample view dropout (independent of CFG)
        drop_view_per_sample = kwargs.get('_drop_view_per_sample')
        if drop_view_per_sample is not None:
            encoded = self._apply_view_mask(encoded, num_views, drop_view_per_sample)

        return encoded

    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference (with frame token and pose embeddings).
        """
        num_views = cond.shape[1] if isinstance(cond, torch.Tensor) and cond.ndim == 5 else 1
        cond = self.encode_image(cond)

        if num_views > 1:
            denoiser = self.models.get('denoiser')
            raw_denoiser = dist_utils.unwrap_dist(denoiser)

            if getattr(raw_denoiser, 'use_frame_token_embedder', False):
                S = cond.shape[1] // num_views
                stacked = cond.reshape(cond.shape[0], num_views, S, cond.shape[2])
                cond = raw_denoiser.encode_frame_tokens(stacked)

            # Apply per-view pose embeddings (additive, like frame tokens)
            pose = kwargs.pop('_pose_for_cond', None)
            if pose is not None and getattr(raw_denoiser, 'use_pose_embedder', False):
                S = cond.shape[1] // num_views
                stacked = cond.reshape(cond.shape[0], num_views, S, cond.shape[2])
                pose_emb = raw_denoiser.encode_pose(pose)  # (B, K, C)
                stacked = stacked + pose_emb[:, :, None, :]
                cond = stacked.reshape(cond.shape[0], num_views * S, cond.shape[2])

        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super(ImageConditionedMixin, self).get_inference_cond(cond, **kwargs)
        return cond

    def _apply_view_mask(self, cond: torch.Tensor, num_views: int, drop_view_per_sample: torch.Tensor) -> torch.Tensor:
        """
        Zero out tokens from dropped views, per-sample.

        Args:
            cond: Encoded conditioning tensor (B, num_views * seq_len, dim)
            num_views: Number of views
            drop_view_per_sample: Tensor of shape (B,) with values:
                -2 = keep all views
                -1 = drop all views (unconditional)
                0..N-1 = drop that specific view

        Returns:
            Masked conditioning tensor with same shape
        """
        if drop_view_per_sample is None:
            return cond

        # Fast path: all samples keep all views
        if (drop_view_per_sample == -2).all():
            return cond

        cond = cond.clone()
        dvps = drop_view_per_sample.to(cond.device)
        seq_per_view = cond.shape[1] // num_views

        # Zero all views for unconditional samples
        uncond = (dvps == -1)
        if uncond.any():
            cond[uncond] = 0

        # Zero specific views
        for v in range(num_views):
            drop_v = (dvps == v)
            if drop_v.any():
                cond[drop_v, v * seq_per_view:(v + 1) * seq_per_view] = 0

        return cond

    def get_pointmap_cond(self, pointmap, **kwargs):
        """
        Get the pointmap conditioning data for multi-view.
        Handles 5D input (B, N, C, H, W) by encoding each view and concatenating
        along the sequence dimension to match image features.
        """
        if pointmap.ndim == 5:
            B, num_views, C, H, W = pointmap.shape

            # Route shared CFG mask expanded to B*N for flattened processing
            shared_mask = kwargs.get('_shared_cfg_mask')
            if shared_mask is not None:
                kwargs['_cfg_mask'] = np.repeat(shared_mask, num_views)  # CFG will pop this

            pointmap_flat = rearrange(pointmap, 'b n c h w -> (b n) c h w')
            cond_flat = super().get_pointmap_cond(pointmap_flat, **kwargs)
            cond = rearrange(cond_flat, '(b n) s d -> b (n s) d', b=B, n=num_views)

            drop_view_per_sample = kwargs.get('_drop_view_per_sample')
            if drop_view_per_sample is not None:
                cond = self._apply_view_mask(cond, num_views, drop_view_per_sample)
            return cond
        else:
            return super().get_pointmap_cond(pointmap, **kwargs)

    def get_mask_cond(self, mask, **kwargs):
        """
        Get the mask conditioning data for multi-view.
        Handles 5D input (B, N, C, H, W) by encoding each view and concatenating
        along the sequence dimension to match image features.
        """
        if mask.ndim == 5:
            B, num_views, C, H, W = mask.shape

            # Route shared CFG mask expanded to B*N (mask follows image)
            shared_mask = kwargs.get('_shared_cfg_mask')
            if shared_mask is not None:
                kwargs['_cfg_mask'] = np.repeat(shared_mask, num_views)  # parent will pop this

            mask_flat = rearrange(mask, 'b n c h w -> (b n) c h w')
            cond_flat = super().get_mask_cond(mask_flat, **kwargs)
            cond = rearrange(cond_flat, '(b n) s d -> b (n s) d', b=B, n=num_views)

            drop_view_per_sample = kwargs.get('_drop_view_per_sample')
            if drop_view_per_sample is not None:
                cond = self._apply_view_mask(cond, num_views, drop_view_per_sample)
            return cond
        else:
            return super().get_mask_cond(mask, **kwargs)

    @torch.no_grad()
    def encode_image(self, images: Union[tuple, list, List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """
        Encode N images and concatenate features along sequence dimension.
        images: tuple/list of N image tensors, single tensor, 5D tensor (B, N, C, H, W), or list of tuples/PIL images
        Returns: [B, N*patch_seq_len, dim] for N images (only patch tokens, no CLS/registers)

        Base encode_image already strips CLS/registers, so we just need to split by view and concatenate.
        """
        # Handle 5D tensor: (B, num_views, C, H, W) -> convert to list of N tensors
        if isinstance(images, torch.Tensor) and images.ndim == 5:
            B, N, C, H, W = images.shape
            images = [images[:, i] for i in range(N)]  # List of N tensors, each (B, C, H, W)

        # Check if it's multi-image conditioning: list/tuple of N tensors
        if (isinstance(images, (tuple, list)) and len(images) >= 1 and
            isinstance(images[0], torch.Tensor) and images[0].ndim == 4):
            # Training: images is [img1_tensor, img2_tensor, ..., imgN_tensor]
            batch_size = images[0].shape[0]
            num_views = len(images)

            # Verify all images have same batch size
            for img in images:
                if img.shape[0] != batch_size:
                    raise ValueError(f"All images must have same batch size, got {img.shape[0]} vs {batch_size}")

            # Parent will concat [img1, img2, ..., imgN] -> (N*B, C, H, W)
            # and return features of shape (N*B, patch_seq_len, dim) - already patch-only
            combined_features = super().encode_image(list(images))

            # Split back into individual image features and concatenate along sequence dim
            # combined_features: (N*B, patch_seq_len, dim) -> (B, N*patch_seq_len, dim)
            combined = rearrange(combined_features, '(n b) s d -> b (n s) d', n=num_views, b=batch_size)
            return combined
            
        elif isinstance(images, torch.Tensor):
            # Single tensor case - treat as regular image conditioning
            # This can happen during negative conditioning in CFG
            return super().encode_image(images)
            
        elif isinstance(images, list):
            # Check if it's a list of tuples (batch of multi-images for inference)
            if len(images) > 0 and isinstance(images[0], tuple):
                # Check if tuples contain tensors or PIL images
                if isinstance(images[0][0], torch.Tensor):
                    # List of (img1_tensor, img2_tensor, ...) tuples
                    n_images = len(images[0])
                    stacked_images = [torch.stack([sample[i] for sample in images]) for i in range(n_images)]
                    # Process as list of N tensors (recursive call)
                    return self.encode_image(stacked_images)
                else:
                    # List of (img1_pil, img2_pil, ...) tuples
                    n_images = len(images[0])
                    image_feats = []
                    for i in range(n_images):
                        imgs = [sample[i] for sample in images]
                        feats = super().encode_image(imgs)
                        image_feats.append(feats)
                    combined = torch.cat(image_feats, dim=1)
                    return combined
            else:
                # List of single PIL images - treat as regular image conditioning
                return super().encode_image(images)
        else:
            raise ValueError(f"MultiImageConditionedMixin expected list of N tensors, single tensor, or list, got {type(images)}")
    
    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data for multi-images.
        Returns all images concatenated horizontally.
        """
        if isinstance(cond, tuple):
            # Concatenate all images side by side along width dimension
            combined = torch.cat(cond, dim=3)  # (B, C, H, N*W)
            return {'image': {'value': combined, 'type': 'image'}}
        elif isinstance(cond, torch.Tensor) and cond.ndim == 5:
            # Multi-view tensor: (B, num_views, C, H, W) -> concatenate views horizontally
            B, N, C, H, W = cond.shape
            combined = rearrange(cond, 'b n c h w -> b c h (n w)')  # (B, C, H, N*W)
            return {'image': {'value': combined, 'type': 'image'}}
        else:
            return super().vis_cond(cond, **kwargs)