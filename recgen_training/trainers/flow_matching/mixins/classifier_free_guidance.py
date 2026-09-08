import torch
import numpy as np
from ....utils.general_utils import dict_foreach
from recgen_inference.recgen_modules.pipelines import samplers


class ClassifierFreeGuidanceMixin:
    def __init__(self, *args, p_uncond: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond

    def get_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"

        # Support pre-computed CFG mask for coordinated dropout across modalities.
        # Always pop to prevent leaking to downstream calls.
        cfg_mask = kwargs.pop('_cfg_mask', None)

        def get_batch_size(cond):
            if isinstance(cond, torch.Tensor):
                return cond.shape[0]
            elif isinstance(cond, (list, tuple)):
                if isinstance(cond, tuple):
                    return get_batch_size(cond[0])
                return len(cond)
            else:
                raise ValueError(f"Unsupported type of cond: {type(cond)}")

        def select(cond, neg_cond, mask):
            if isinstance(cond, torch.Tensor):
                mask_tensor = torch.tensor(mask, device=cond.device).reshape(-1, *[1] * (cond.ndim - 1))
                return torch.where(mask_tensor, neg_cond, cond)
            elif isinstance(cond, tuple):
                return tuple(select(c, nc, mask) for c, nc in zip(cond, neg_cond))
            elif isinstance(cond, list):
                return [nc if m else c for c, nc, m in zip(cond, neg_cond, mask)]
            else:
                raise ValueError(f"Unsupported type of cond: {type(cond)}")

        if cfg_mask is not None:
            mask = list(cfg_mask)
        elif self.p_uncond > 0:
            ref_cond = cond if not isinstance(cond, dict) else cond[list(cond.keys())[0]]
            B = get_batch_size(ref_cond)
            mask = list(np.random.rand(B) < self.p_uncond)
        else:
            return cond

        if not isinstance(cond, dict):
            cond = select(cond, neg_cond, mask)
        else:
            cond = dict_foreach([cond, neg_cond], lambda x: select(x[0], x[1], mask))

        return cond

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data for inference.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"
        return {'cond': cond, 'neg_cond': neg_cond, **kwargs}
    
    def get_sampler(self, **kwargs) -> samplers.FlowEulerCfgSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerCfgSampler(self.sigma_min)
