"""Inference entry points: generate (single-view) and generate_multiview."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from recgen_inference._result import RecGenResult, build_result
from recgen_inference.preprocessing import (
    normalize_depth,
    preprocess_view,
)
from recgen_inference.utils import mesh_from_result, parse_pose


def _preprocess_single(
    image: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    quantile_drop_threshold: float,
    clamp_range: Tuple[float, float],
    mask_erosion_enabled: bool,
    mask_erosion_params: Optional[Dict[str, int]],
) -> Dict[str, Any]:
    """Preprocess one view into tensors the pipeline can consume."""
    return preprocess_view(
        image,
        normalize_depth(depth),
        mask,
        intrinsics,
        quantile_drop_threshold=quantile_drop_threshold,
        clamp_range=clamp_range,
        mask_erosion_enabled=mask_erosion_enabled,
        mask_erosion_params=mask_erosion_params,
    )


def _build_result_from_outputs(
    outputs: Dict[str, Any],
    cam2ncam: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    posthoc_color: str = "none",
    mask_full: Optional[np.ndarray] = None,
) -> RecGenResult:
    """Shared postprocessing path: parse pose, build mesh, transform into camera frame."""
    pose_matrix, parsed_pose, pose_representation = parse_pose(outputs)

    # Post-hoc color calibration of the asset against the input photo
    # (HB-validated: gamma_affine improves novel-view color L1 by ~25% at ~26 ms).
    # If the fit gates out, the asset is left untouched (identity).
    posthoc_info = None
    if posthoc_color and posthoc_color != "none":
        from recgen_inference.recgen_modules.utils import posthoc_color as _pcc

        tr, st = None, {"reason": "no gaussian output"}
        if "gaussian" in outputs and mask_full is not None:
            # canonical -> input-camera similarity (undo the normalization)
            M = np.linalg.inv(np.asarray(cam2ncam, np.float64)) @ pose_matrix
            tr, st = _pcc.calibrate_asset(
                outputs["gaussian"][0], M, np.asarray(intrinsics),
                np.asarray(rgb), mask_full, variant=posthoc_color,
            )
        if tr is not None:
            import torch

            _pcc.apply_to_gaussian(outputs["gaussian"][0], tr)
            m = outputs["mesh"][0]
            if getattr(m, "vertex_attrs", None) is not None:
                c = m.vertex_attrs[:, :3].clamp(0, 1).detach().cpu().numpy()
                m.vertex_attrs[:, :3] = torch.tensor(
                    _pcc.apply_transform(c, tr),
                    dtype=m.vertex_attrs.dtype, device=m.vertex_attrs.device,
                )
            print(f"[recgen_inference] post-hoc color ({posthoc_color}): applied "
                  f"(overlap {st['overlap_px']}px, iou {st['iou']:.2f}, "
                  f"fit L1 {st['l1_before']:.3f} -> {st['l1_after']:.3f})")
        else:
            print(f"[recgen_inference] post-hoc color ({posthoc_color}): identity ({st.get('reason')})")
        posthoc_info = {"variant": posthoc_color, "transform": tr, "stats": st}

    raw_trimesh = mesh_from_result(outputs["mesh"][0])

    final_mesh = raw_trimesh.copy()
    final_mesh.apply_transform(pose_matrix)
    final_mesh.apply_translation(-cam2ncam[:3, 3])
    final_mesh.apply_scale(1.0 / cam2ncam[0, 0])

    return build_result(
        mesh=final_mesh,
        raw_mesh=raw_trimesh,
        pose_matrix=pose_matrix,
        parsed_pose=parsed_pose,
        cam2ncam=cam2ncam,
        pose_representation=pose_representation,
        outputs=outputs,
        rgb=rgb,
        intrinsics=intrinsics,
        posthoc_color=posthoc_info,
    )


def generate(
    pipeline: Any,
    image: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    seed: int = 1,
    quantile_drop_threshold: float = 0.05,
    clamp_range: Tuple[float, float] = (-2.0, 3.0),
    mask_erosion_enabled: bool = True,
    mask_erosion_params: Optional[Dict[str, int]] = None,
    use_pointmap: bool = True,
    posthoc_color: str = "none",
) -> RecGenResult:
    """Generate a 3D mesh from a single RGB-D view.

    Args:
        pipeline: Pipeline from :func:`recgen_inference.build_recgen.build`.
        image: (H, W, 3) uint8 RGB numpy array.
        depth: (H, W) depth map. Accepts uint16 mm or float32 m — the unit is
            auto-detected by :func:`preprocessing.normalize_depth`.
        mask: (H, W) object mask. Non-zero pixels mark the object.
        intrinsics: (3, 3) camera intrinsic matrix.
        seed: Random seed.
        quantile_drop_threshold: Fraction of points to drop from each end when
            computing the robust unit-cube fit.
        clamp_range: Range into which the normalised pointmap is clamped.
        mask_erosion_enabled: Whether to erode the mask before preprocessing.
        mask_erosion_params: Optional ``{"kernel_size": int, "iterations": int}`` override.
        use_pointmap: If False, the pipeline runs without the pointmap branch.
        posthoc_color: ``'none' | 'gamma' | 'affine' | 'gamma_affine'`` — post-hoc
            color calibration of the generated asset against the input photo.
            The predicted Gaussians are rendered into the input view and a
            color transform is fitted on the mask∩render overlap; if the fit
            passes the quality gates it is applied to the Gaussian DC colors
            and mesh vertex colors (else the output is unchanged). The fitted
            transform and stats are recorded in ``result.posthoc_color`` and
            written to ``posthoc_color.json`` on save.
    """
    proc = _preprocess_single(
        image,
        depth,
        mask,
        intrinsics,
        quantile_drop_threshold=quantile_drop_threshold,
        clamp_range=clamp_range,
        mask_erosion_enabled=mask_erosion_enabled,
        mask_erosion_params=mask_erosion_params,
    )

    outputs = pipeline.run_pointmap(
        proc["image"],
        pointmap=proc["pointmap"] if use_pointmap else None,
        mask=proc["mask"],
        seed=seed,
    )

    return _build_result_from_outputs(
        outputs, proc["cam2ncam"], rgb=image, intrinsics=intrinsics,
        posthoc_color=posthoc_color, mask_full=proc.get("mask_full"),
    )


def generate_multiview(
    pipeline: Any,
    anchor_view: Dict[str, Any],
    second_views: List[Dict[str, Any]],
    *,
    seed: int = 1,
    quantile_drop_threshold: float = 0.05,
    clamp_range: Tuple[float, float] = (-2.0, 3.0),
    mask_erosion_enabled: bool = True,
    mask_erosion_params: Optional[Dict[str, int]] = None,
    use_pointmap: bool = True,
    multidiffusion_mode: str = "multidiffusion",
    slat_multiview_mode: Optional[str] = None,
    posthoc_color: str = "none",
) -> RecGenResult:
    """Generate a 3D mesh from an anchor view + N supporting views.

    Each view dict must contain:
        - ``rgb``: (H, W, 3) uint8
        - ``depth``: (H, W) depth (uint16 mm or float32 m — auto-detected)
        - ``mask``: (H, W) object mask
        - ``camera_intrinsics``: (3, 3)

    The result is expressed in the anchor view's camera frame.

    With one second view this runs the plain 2-view pipeline (matching
    training). With N > 1 second views the model stays in its 2-view regime by
    fusing the (anchor, view_i) conditioning PAIRS at every denoising step of
    the sparse-structure stage:

        - ``multidiffusion_mode='multidiffusion'``: average the N velocity
          predictions per Euler step (N forwards/step, best quality — on ABO,
          12-view fusion reduces SLAT latent MSE ~19% vs 2-view).
        - ``multidiffusion_mode='stochastic'``: round-robin one pair per step
          (no extra compute; ~13% on the same benchmark). Prefer this in
          latency-sensitive paths.

    ``slat_multiview_mode`` (None | 'multidiffusion' | 'stochastic') optionally
    applies the same pair fusion to the SLAT (appearance) stage. Each extra
    pair's pose is predicted with an additional sparse-structure pose pass,
    with the anchor pose token pinned from the first pair so all packets agree
    where the anchor camera sits. None keeps the default of anchor + first
    second view (the exact 2-view training regime).
    """
    if len(second_views) < 1:
        raise ValueError("generate_multiview needs at least one second view")

    proc1 = preprocess_view(
        anchor_view["rgb"],
        normalize_depth(anchor_view["depth"]),
        anchor_view["mask"],
        anchor_view["camera_intrinsics"],
        quantile_drop_threshold=quantile_drop_threshold,
        clamp_range=clamp_range,
        mask_erosion_enabled=mask_erosion_enabled,
        mask_erosion_params=mask_erosion_params,
    )
    procs2 = [
        preprocess_view(
            v["rgb"],
            normalize_depth(v["depth"]),
            v["mask"],
            v["camera_intrinsics"],
            quantile_drop_threshold=quantile_drop_threshold,
            clamp_range=clamp_range,
            mask_erosion_enabled=mask_erosion_enabled,
            mask_erosion_params=mask_erosion_params,
        )
        for v in second_views
    ]

    if len(procs2) == 1:
        # Plain 2-view inference (matches 2-view training exactly)
        all_procs = [proc1] + procs2
        outputs = pipeline.run_pointmap_multiview(
            images=[p["image"] for p in all_procs],
            pointmaps=[p["pointmap"] for p in all_procs] if use_pointmap else None,
            masks=[p["mask"] for p in all_procs],
            seed=seed,
        )
    else:
        # N-view: fuse (anchor, view_i) pairs at every denoising step so the
        # 2-view model never sees more than 2 views at once.
        outputs = pipeline.run_pointmap_many_views(
            view1_image=proc1["image"],
            views2_images=[p["image"] for p in procs2],
            view1_pointmap=proc1["pointmap"] if use_pointmap else None,
            views2_pointmaps=[p["pointmap"] for p in procs2] if use_pointmap else None,
            view1_mask=proc1["mask"],
            views2_masks=[p["mask"] for p in procs2],
            seed=seed,
            multidiffusion_mode=multidiffusion_mode,
            slat_multiview_mode=slat_multiview_mode,
        )

    return _build_result_from_outputs(
        outputs,
        proc1["cam2ncam"],
        rgb=anchor_view["rgb"],
        intrinsics=anchor_view["camera_intrinsics"],
        posthoc_color=posthoc_color,
        mask_full=proc1.get("mask_full"),
    )
