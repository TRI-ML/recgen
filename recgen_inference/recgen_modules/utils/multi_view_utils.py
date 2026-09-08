"""
Multi-view diffusion sampling utilities.

Provides a context manager that temporarily patches a FlowEulerCfgSampler's
_inference_model to fuse predictions from multiple second-view conditions
at every denoising step.

Two fusion modes:
  multidiffusion — Average velocity predictions from all N second views.
                   N model forward passes (each with CFG) per Euler step.
  stochastic     — Round-robin: pick one view per step (no overhead per step).
"""

import warnings
from contextlib import contextmanager

import torch


@contextmanager
def multidiffusion_sampling(sampler, view_conds, num_steps=50, mode="multidiffusion"):
    """
    Context manager: patches sampler._inference_model for multi-view sampling.

    The sampler's normal call path passes a single 'cond' tensor to
    _inference_model.  This patch intercepts that call and replaces the
    single condition with N view conditions, fusing the resulting velocity
    predictions before returning.

    Args:
        sampler:    FlowEulerCfgSampler instance (pipeline.sparse_structure_sampler).
        view_conds: List of N conditioning tensors, shape (1, 2*seq_len, dim).
                    Each entry is get_cond_multiview([view1, view2_i])['cond'].
        num_steps:  Total Euler steps (used to pre-compute stochastic schedule).
        mode:       'multidiffusion' (default) or 'stochastic'.

    Yields:
        Nothing — used purely for side effects (patch / restore).

    Example:
        view_conds = [c['cond'] for c in conds_list]
        with multidiffusion_sampling(pipeline.sparse_structure_sampler, view_conds, steps):
            coords, pose_norm, pose = pipeline.sample_sparse_structure_pose(conds_list[0], ...)
    """
    original_inference = sampler._inference_model  # bound method (includes CFG)
    num_views = len(view_conds)

    if num_views == 0:
        raise ValueError("view_conds must contain at least one condition tensor.")

    if mode == "stochastic":
        if num_views > num_steps:
            warnings.warn(
                f"multidiffusion_sampling: num_views ({num_views}) > num_steps "
                f"({num_steps}); some views will never be used."
            )
        # Pre-compute round-robin schedule: step i uses view i % num_views
        cond_schedule = [view_conds[i % num_views] for i in range(num_steps)]
        step_counter = [0]  # mutable int via list so closure can mutate it

        def _new_inference(model, x_t, t, cond, **kwargs):
            idx = min(step_counter[0], len(cond_schedule) - 1)
            vc = cond_schedule[idx]
            step_counter[0] += 1
            return original_inference(model, x_t, t, vc, **kwargs)

    else:  # multidiffusion (default)
        def _fuse(preds):
            # SparseTensor predictions (SLAT stage) can't torch.stack — average feats.
            if hasattr(preds[0], 'replace') and hasattr(preds[0], 'feats'):
                return preds[0].replace(torch.stack([p.feats for p in preds]).mean(0))
            return torch.stack(preds).mean(0)

        def _new_inference(model, x_t, t, cond, **kwargs):
            # 'cond' (from the normal sampler call) is intentionally ignored;
            # we use view_conds from the closure instead.
            preds = []
            pose_preds = []
            for vc in view_conds:
                pred, pose_pred = original_inference(model, x_t, t, vc, **kwargs)
                preds.append(pred)
                if pose_pred is not None:
                    pose_preds.append(pose_pred)

            avg_pred = _fuse(preds)
            avg_pose_pred = torch.stack(pose_preds).mean(0) if pose_preds else None
            return avg_pred, avg_pose_pred

    # Patch the instance (shadows the class method via instance __dict__)
    sampler._inference_model = _new_inference
    try:
        yield
    finally:
        # Always restore — even if inference raises
        sampler._inference_model = original_inference
