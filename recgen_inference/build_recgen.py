"""Build pre-trained RecGen pipelines.

Design inspired by timm. Users pick a model name from
`list_models()` or `list_checkpoints()` and call `build(name, pretrained=True)`
to get a ready-to-run pipeline.

Example
-------
>>> from recgen_inference import build_recgen
>>> print(build_recgen.list_checkpoints())
['recgen_base.multiview_stereo']
>>> pipeline = build_recgen.build("recgen_base.multiview_stereo")
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional

import torch

# Default HuggingFace repo (self-contained: also mirrors TRELLIS base model files)
DEFAULT_HF_REPO: Final[str] = "TRI-ML/RecGen"


def _cfg(
    hf_subdir: str,
    slat_ckpt: str = "slat_denoiser_ema0.9999_step0075000.pt",
    sparse_ckpt: str = "stereo_denoiser_ema0.9999_step0055000.pt",
    slat_config: str = "slat_config.json",
    sparse_config: str = "stereo_config.json",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Return the default configuration for a RecGen model.

    The config describes where to find the RecGen-specific weights + configs
    inside `DEFAULT_HF_REPO`. All models share the same base TRELLIS pipeline.
    """
    return {
        "hf_repo": DEFAULT_HF_REPO,
        "hf_subdir": hf_subdir,
        "slat_ckpt": slat_ckpt,
        "sparse_ckpt": sparse_ckpt,
        "slat_config": slat_config,
        "sparse_config": sparse_config,
        **kwargs,
    }


# Model configs. Keyed as "<model-architecture>.<checkpoint>".
CONFIGS: Final[Dict[str, Dict[str, Any]]] = {
    # Stage 1 = the paper model fine-tuned with ColorJitter and
    # SAM2-predicted-mask mixing for robustness to real-world lighting and
    # imperfect segmentation masks; stage 2 (SLAT) = the paper checkpoint.
    # The paper's stage-1 weights (step 55k) remain on the hub —
    # recgen_eval pins them explicitly to reproduce the published table.
    "recgen_base.multiview_stereo": _cfg(
        hf_subdir="",
        sparse_ckpt="sparse-structure-ft-70k/stereo_denoiser_ema0.9999_step0070000.pt",
    ),
}


# Registry for builder functions.
_ENTRY_POINTS: Dict[str, Callable[..., Any]] = {}


def register_model() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for registering builder functions."""

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        assert func.__name__.startswith("build_")
        model_name = func.__name__.replace("build_", "", 1)
        assert model_name not in _ENTRY_POINTS
        _ENTRY_POINTS[model_name] = func
        return func

    return wrapper


def list_models() -> List[str]:
    """Return registered model architectures (without checkpoint suffixes)."""
    return sorted(_ENTRY_POINTS.keys())


def list_checkpoints() -> List[str]:
    """Return all known `<architecture>.<checkpoint>` keys."""
    return sorted(CONFIGS.keys())


def get_config(model_name: str) -> Optional[Dict[str, Any]]:
    """Resolve `<architecture>[.<checkpoint>]` into its CONFIGS entry."""
    model_base, _, weights = model_name.partition(".")
    if weights == "":
        avail = [k for k in CONFIGS if k.partition(".")[0] == model_base]
        if not avail:
            return None
        weights = avail[0].partition(".")[-1]
    key = f"{model_base}.{weights}" if weights else model_base
    return CONFIGS.get(key)


def _load_model_config(
    checkpoint_path: str,
    hf_config_name: str,
    repo_id: str,
    repo_subdir: str,
) -> Optional[Dict[str, Any]]:
    """Look for a model-config JSON next to the checkpoint, then fall back to HF."""
    ckpt_dir = Path(checkpoint_path).parent
    named = ckpt_dir / hf_config_name
    if named.exists():
        with open(named, "r") as f:
            return json.load(f)

    for parent in (ckpt_dir, ckpt_dir.parent, ckpt_dir.parent.parent):
        cfg_path = parent / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                return json.load(f)

    try:
        from huggingface_hub import hf_hub_download

        cfg_file = hf_hub_download(repo_id, f"{repo_subdir}/{hf_config_name}" if repo_subdir else hf_config_name)
        with open(cfg_file, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _load_pose_stats(
    stats_filename: str,
    checkpoint_path: str,
    repo_id: str,
    repo_subdir: str,
) -> Any:
    """Load pose normalization stats from a local path or HuggingFace."""
    from recgen_inference.recgen_modules.utils.pose_utils import (
        PoseNormalizer,
        load_pose_normalization_stats,
    )

    local = Path(checkpoint_path).parent / stats_filename
    if local.exists():
        stats = load_pose_normalization_stats(str(local))
        return PoseNormalizer(stats)

    from huggingface_hub import hf_hub_download

    sf = hf_hub_download(repo_id, f"{repo_subdir}/{stats_filename}" if repo_subdir else stats_filename)
    stats = load_pose_normalization_stats(sf)
    return PoseNormalizer(stats)


@register_model()
def build_recgen_base(
    pretrained: bool = True,
    *,
    config: Optional[Dict[str, Any]] = None,
    checkpoint_slat: Optional[str] = None,
    checkpoint_sparse: Optional[str] = None,
    device: str = "cuda",
) -> Any:
    """Build the base RecGen pipeline and load weights.

    Args:
        pretrained: If True, auto-download missing checkpoints from HuggingFace.
        config: Optional CONFIGS dict override (e.g. pointing at a different HF subdir).
        checkpoint_slat: Override path to the SLAT denoiser checkpoint.
        checkpoint_sparse: Override path to the sparse-structure denoiser checkpoint.
        device: Device to move the pipeline to. Defaults to "cuda".

    Returns:
        A configured `RecGenPipeline` ready for `run_pointmap(...)` /
        `run_pointmap_multiview(...)` calls.
    """
    if config is None:
        config = CONFIGS["recgen_base.multiview_stereo"]

    from huggingface_hub import hf_hub_download

    from recgen_inference.recgen_modules import models
    from recgen_inference.recgen_modules.pipelines import (
        RecGenPipeline,
        samplers,
    )

    repo_id = config["hf_repo"]
    repo_subdir = config["hf_subdir"]
    sub = lambda p: f"{repo_subdir}/{p}" if repo_subdir else p
    # Configs and pose stats live NEXT TO their checkpoint in the repo: a
    # checkpoint stored in a subdirectory (e.g. "sparse-structure-ft-70k/...")
    # brings its own stereo_config.json / stereo_pose_stats.json, which may
    # differ from the root (paper) ones.
    import posixpath

    sparse_repo_dir = posixpath.dirname(sub(config["sparse_ckpt"]))
    slat_repo_dir = posixpath.dirname(sub(config["slat_ckpt"]))

    print("[recgen_inference] Loading pipeline...")

    if checkpoint_slat is None:
        if not pretrained:
            raise ValueError("checkpoint_slat is required when pretrained=False")
        checkpoint_slat = hf_hub_download(repo_id, sub(config["slat_ckpt"]))
    if checkpoint_sparse is None:
        if not pretrained:
            raise ValueError("checkpoint_sparse is required when pretrained=False")
        checkpoint_sparse = hf_hub_download(repo_id, sub(config["sparse_ckpt"]))

    config_file = hf_hub_download(repo_id, sub("pipeline.json"))
    with open(config_file, "r") as f:
        base_config = json.load(f)
    args = base_config["args"]

    # Load base models (skip the two we replace)
    _models: Dict[str, Any] = {}
    for k, v in args["models"].items():
        if k not in ["sparse_structure_flow_model", "slat_flow_model"]:
            _models[k] = models.from_pretrained(f"{repo_id}/{sub(v)}")

    pipeline = RecGenPipeline(
        models=_models,
        sparse_structure_sampler=None,
        slat_sampler=None,
        slat_normalization=None,
        image_cond_model=args["image_cond_model"],
    )
    pipeline._pretrained_args = args

    pipeline.sparse_structure_sampler = getattr(
        samplers, args["sparse_structure_sampler"]["name"]
    )(**args["sparse_structure_sampler"]["args"])
    pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]
    pipeline.slat_sampler = getattr(samplers, args["slat_sampler"]["name"])(
        **args["slat_sampler"]["args"]
    )
    pipeline.slat_sampler_params = args["slat_sampler"]["params"]

    # Sparse-structure + pose model
    sparse_config = _load_model_config(
        checkpoint_sparse, config["sparse_config"], repo_id, sparse_repo_dir
    )
    if sparse_config and "models" in sparse_config and "denoiser" in sparse_config["models"]:
        sparse_model_config = sparse_config["models"]["denoiser"]["args"]
    else:
        sparse_model_config = {
            "resolution": 16, "in_channels": 8, "out_channels": 8,
            "model_channels": 1024, "cond_channels": 1024, "num_blocks": 24,
            "num_heads": 16, "mlp_ratio": 4, "patch_size": 1, "pe_mode": "ape",
            "qk_rms_norm": True, "use_fp16": True,
            "use_point_embedder": True, "point_embedder_out_channels": 1024,
            "use_mask_embedder": True, "mask_embedder_out_channels": 1024,
        }
    pipeline.models["sparse_structure_pose_flow_model"] = models.SparseStructurePoseFlowModel(
        **sparse_model_config
    )

    # Pose normalizer
    pose_normalizer = None
    pose_representation = None
    if sparse_config and "trainer" in sparse_config:
        trainer_cfg = sparse_config["trainer"].get("args", sparse_config["trainer"])
        use_pose_norm = trainer_cfg.get("use_pose_normalization", False)
        pose_representation = trainer_cfg.get("pose_representation", None)
        if use_pose_norm:
            norm_path = trainer_cfg.get("pose_normalization_config", None)
            if norm_path and os.path.exists(norm_path):
                from recgen_inference.recgen_modules.utils.pose_utils import (
                    PoseNormalizer,
                    load_pose_normalization_stats,
                )

                stats = load_pose_normalization_stats(norm_path)
                pose_normalizer = PoseNormalizer(stats)
            else:
                pose_normalizer = _load_pose_stats(
                    "stereo_pose_stats.json", checkpoint_sparse, repo_id, sparse_repo_dir
                )

    pipeline.pose_normalizer = pose_normalizer
    pipeline.pose_representation = pose_representation

    # SLAT model
    slat_config = _load_model_config(
        checkpoint_slat, config["slat_config"], repo_id, slat_repo_dir
    )
    if slat_config and "models" in slat_config and "denoiser" in slat_config["models"]:
        slat_model_config = slat_config["models"]["denoiser"]["args"]
    else:
        slat_model_config = {
            "resolution": 64, "in_channels": 8, "out_channels": 8,
            "model_channels": 1024, "cond_channels": 1024, "num_blocks": 24,
            "num_heads": 16, "mlp_ratio": 4, "patch_size": 2,
            "num_io_res_blocks": 2, "io_block_channels": [128], "pe_mode": "ape",
            "qk_rms_norm": True, "use_fp16": True,
            "use_point_embedder": True, "point_embedder_out_channels": 1024,
            "use_mask_embedder": True, "mask_embedder_out_channels": 1024,
            "use_pose_embedder": False,
        }
    pipeline.models["slat_flow_model"] = models.SLatCondFlowModel(**slat_model_config)

    pipeline.slat_normalization = (
        slat_config.get("dataset", {}).get("args", {}).get("normalization", None)
        if slat_config
        else None
    )

    slat_flow = pipeline.models["slat_flow_model"]
    pipeline.slat_use_pose = getattr(slat_flow, "use_pose_embedder", False)
    pipeline.slat_pose_representation = getattr(slat_flow, "pose_representation", None)

    pipeline.slat_pose_normalizer = None
    if pipeline.slat_use_pose and slat_config:
        slat_dataset = slat_config.get("dataset", {}).get("args", {})
        if slat_dataset.get("use_pose_normalization", False):
            pipeline.slat_pose_normalizer = _load_pose_stats(
                "slat_pose_stats.json", checkpoint_slat, repo_id, slat_repo_dir
            )

    # Load weights
    slat_weights = torch.load(checkpoint_slat, map_location="cpu")
    pipeline.models["slat_flow_model"].load_state_dict(slat_weights, strict=False)

    sparse_weights = torch.load(checkpoint_sparse, map_location="cpu")
    pipeline.models["sparse_structure_pose_flow_model"].load_state_dict(
        sparse_weights, strict=False
    )

    # Multi-view detection
    trainer_name = (sparse_config or {}).get("trainer", {}).get("name", "")
    dataset_nv = (
        (sparse_config or {}).get("dataset", {}).get("args", {}).get("num_views", 1)
    )
    pipeline.is_multiview = "MultiImage" in trainer_name or dataset_nv > 1

    pipeline.to(device)
    print("[recgen_inference] Pipeline loaded.")
    return pipeline


def build(
    model_name: str = "recgen_base.multiview_stereo",
    pretrained: bool = True,
    *,
    checkpoint_slat: Optional[str] = None,
    checkpoint_sparse: Optional[str] = None,
    device: str = "cuda",
) -> Any:
    """Build a RecGen pipeline by name.

    Args:
        model_name: Key into CONFIGS, e.g. ``"recgen_base.multiview_stereo"``. The
            architecture part alone (``"recgen_base"``) picks the first matching
            checkpoint.
        pretrained: If True, auto-download missing checkpoints from HuggingFace.
        checkpoint_slat: Local override for the SLAT denoiser weights.
        checkpoint_sparse: Local override for the sparse-structure weights.
        device: Device to move the pipeline to.
    """
    config = get_config(model_name)
    if config is None:
        raise ValueError(
            f"Unknown model {model_name!r}. Known checkpoints: {list_checkpoints()}"
        )
    arch = model_name.partition(".")[0]
    if arch not in _ENTRY_POINTS:
        raise ValueError(f"No builder registered for architecture {arch!r}")
    builder = _ENTRY_POINTS[arch]
    return builder(
        pretrained=pretrained,
        config=config,
        checkpoint_slat=checkpoint_slat,
        checkpoint_sparse=checkpoint_sparse,
        device=device,
    )
