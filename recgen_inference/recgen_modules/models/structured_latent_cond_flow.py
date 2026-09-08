from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules.norm import LayerNorm32
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
from .sparse_structure_flow import TimestepEmbedder
from .sparse_elastic_mixin import SparseTransformerElasticMixin


POSE_REPRESENTATION_DIMENSIONS = {
    "quaternion_translation_scale": 8,
    "6d_translation_scale": 10,
    "9d_translation_scale": 13,
}

class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, bias=True),
        )
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = None
        if self.downsample:
            self.updown = sp.SparseDownsample(2)
        elif self.upsample:
            self.updown = sp.SparseUpsample(2)

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.updown is not None:
            x = self.updown(x)
        return x

    def forward(self, x: sp.SparseTensor, emb: torch.Tensor) -> sp.SparseTensor:
        emb_out = self.emb_layers(emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        x = self._updown(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)

        return h
    


class SLatCondFlowModel(nn.Module):
    """
    SLat Flow Model with additional conditioning support via pointmap and mask embedders.
    Used for overlay conditioning in part-aware 3D generation.
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        use_point_embedder: bool = False,
        point_embedder_out_channels: int = 1024,
        use_mask_embedder: bool = False,
        mask_embedder_out_channels: int = 1024,
        use_pose_embedder: bool = False,
        pose_embedder_out_channels: int = 1024,
        pose_representation: Literal["quaternion_translation_scale", "6d_translation_scale", "9d_translation_scale"] = "quaternion_translation_scale",
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_io_res_blocks = num_io_res_blocks
        self.io_block_channels = io_block_channels
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        # Embedder configuration
        self.use_point_embedder = use_point_embedder
        self.point_embedder_out_channels = point_embedder_out_channels
        self.use_mask_embedder = use_mask_embedder
        self.mask_embedder_out_channels = mask_embedder_out_channels
        self.use_pose_embedder = use_pose_embedder
        self.pose_embedder_out_channels = pose_embedder_out_channels
        self.pose_representation = pose_representation


        if self.io_block_channels is not None:
            assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"
            assert np.log2(patch_size) == len(io_block_channels), "Number of IO ResBlocks must match the number of stages"

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels if io_block_channels is None else io_block_channels[0])
        
        self.input_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
                self.input_blocks.extend([
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks-1)
                ])
                self.input_blocks.append(
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=next_chs,
                        downsample=True,
                    )
                )
            
        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, prev_chs in zip(reversed(io_block_channels), [model_channels] + list(reversed(io_block_channels[1:]))):
                self.out_blocks.append(
                    SparseResBlock3d(
                        prev_chs * 2 if self.use_skip_connection else prev_chs,
                        model_channels,
                        out_channels=chs,
                        upsample=True,
                    )
                )
                self.out_blocks.extend([
                    SparseResBlock3d(
                        chs * 2 if self.use_skip_connection else chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks-1)
                ])
            
        self.out_layer = sp.SparseLinear(model_channels if io_block_channels is None else io_block_channels[0], out_channels)
        
        # Add optional point embedder for pointmap conditioning
        # Conv2d with kernel_size=14, stride=14 to match DINOv2 patch size
        if self.use_point_embedder:
            self.point_embedder = nn.Conv2d(3, self.point_embedder_out_channels, kernel_size=14, stride=14)
        
        # Add optional mask embedder for mask conditioning
        if self.use_mask_embedder:
            self.mask_embedder = nn.Conv2d(1, self.mask_embedder_out_channels, kernel_size=14, stride=14)
        
        # Add optional pose embedder for pose conditioning
        # MLP to embed pose representation to feature dimension
        if self.use_pose_embedder:
            self.pose_embedder = nn.Linear(POSE_REPRESENTATION_DIMENSIONS[self.pose_representation], self.pose_embedder_out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.blocks.apply(convert_module_to_f32)
        self.out_blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)
        
        # Initialize point embedder if present
        if self.use_point_embedder:
            nn.init.xavier_uniform_(self.point_embedder.weight)
            if self.point_embedder.bias is not None:
                nn.init.constant_(self.point_embedder.bias, 0)
        
        # Initialize mask embedder if present
        if self.use_mask_embedder:
            nn.init.xavier_uniform_(self.mask_embedder.weight)
            if self.mask_embedder.bias is not None:
                nn.init.constant_(self.mask_embedder.bias, 0)
        
        # Initialize pose embedder if present
        if self.use_pose_embedder:
            nn.init.xavier_uniform_(self.pose_embedder.weight)
            if self.pose_embedder.bias is not None:
                nn.init.constant_(self.pose_embedder.bias, 0)

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, **kwargs) -> sp.SparseTensor:
        h = self.input_layer(x).type(self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        skips = []
        # pack with input blocks
        for block in self.input_blocks:
            h = block(h, t_emb)
            skips.append(h.feats)
        
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)

        # unpack with output blocks
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(x.dtype))
        return h
    
    def encode_pointmap(self, pointmap: torch.Tensor) -> torch.Tensor:
        """
        Encode pointmap using the point embedder.
        
        Args:
            pointmap: Pointmap tensor of shape (B, 3, H, W)
            
        Returns:
            Encoded pointmap features of shape (B, N, C) where N = (H/14) * (W/14)
        """
        if not self.use_point_embedder:
            raise ValueError("Point embedder is not enabled for this model. Set use_point_embedder=True during initialization.")
        
        features = self.point_embedder(pointmap)  # (B, C, H', W')
        features = features.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        return features
    
    def encode_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Encode mask using the mask embedder.
        
        Args:
            mask: Mask tensor of shape (B, 1, H, W)
            
        Returns:
            Encoded mask features of shape (B, N, C) where N = (H/14) * (W/14)
        """
        if not self.use_mask_embedder:
            raise ValueError("Mask embedder is not enabled for this model. Set use_mask_embedder=True during initialization.")
        
        features = self.mask_embedder(mask)  # (B, C, H', W')
        features = features.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        return features
    
    def encode_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Encode pose using the pose embedder.

        Args:
            pose: Pose tensor of shape (B, D) for single pose or (B, K, D) for K poses.
                  D=8 for quaternion or D=13 for rotation matrix representation.

        Returns:
            Encoded pose features of shape (B, K, C) for concatenation as conditioning tokens.
            For single pose input (B, D), returns (B, 1, C).
        """
        if not self.use_pose_embedder:
            raise ValueError("Pose embedder is not enabled for this model. Set use_pose_embedder=True during initialization.")

        if pose.shape[-1] not in (8, 13):
            raise ValueError(f"Pose must be 8D (quaternion) or 13D (rotation matrix), got shape {pose.shape}")

        # nn.Linear applies to last dimension, works for both (B, D) and (B, K, D)
        features = self.pose_embedder(pose)
        if features.ndim == 2:
            # Single pose: (B, C) -> (B, 1, C)
            features = features.unsqueeze(1)
        # Multi-pose: (B, K, C) already has the sequence dimension
        return features


class ElasticSLatCondFlowModel(SparseTransformerElasticMixin, SLatCondFlowModel):
    """
    SLat Cond Flow Model with elastic memory management.
    Used for training with low VRAM.
    """
    pass