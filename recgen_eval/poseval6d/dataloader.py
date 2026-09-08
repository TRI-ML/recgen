from .datasets.multiview_anchor_dataset import MultiViewAnchorDataset
from .datasets.bop_dataset import HBDataset
from .datasets.bop_dataset import AV2Dataset


def load_multiview_anchor_dataset(
    dataset_name: str,
    path_to_datasets: str,
    save_path: str,
    num_views: int = 2,
    split_size: int = 0,
    split_index: int = 0,
    subsample_step: int = 0,
    debug: bool = False,
    instance_list_path: str = None,
    hb_test_subdir: str = "val_kinect",
):
    """
    Load multi-view anchor dataset for evaluation.

    For multi-view evaluation, we always load the base dataset with subsample_step=1
    because the instance_list.txt defines specific frame pairs that need to be available.
    Debug mode is handled by limiting the number of pairs evaluated (via max_samples),
    not by subsampling the base dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "HB")
        path_to_datasets: Root path to datasets
        save_path: Path to save generated meshes
        num_views: Number of views per sample
        split_size: Total number of splits for parallel evaluation
        split_index: Index of this split
        subsample_step: Subsampling step for the base dataset (ignored for multi-view, always 1)
        debug: Whether to use debug mode (fewer samples) - controlled via max_samples, not subsampling
        instance_list_path: Custom path to instance_list.txt (optional)

    Returns:
        Tuple of (MultiViewAnchorDataset, BaseDataset)
    """
    # For multi-view, always use subsample_step=1 since instance_list.txt defines specific frame pairs
    # Debug mode is handled by max_samples in the evaluation script, not by dataset subsampling
    step = 1
    if debug:
        print(f"Multi-view mode: Loading ALL frames (subsample_step=1) to match instance_list.txt pairs.")
        print(f"Debug limiting will be done via --max_samples argument instead.")

    # Load the base dataset with subsample_step=1
    if dataset_name == "HB":
        # For multi-view, load all frames so second views at any frame can be found
        dataset = HBDataset(root=f"{path_to_datasets}/{dataset_name}", test_subdir=hb_test_subdir, resize_factor=0.5, subsample_step=step, load_all_frames=True)
    elif dataset_name in ("AV2", "AV_final"):
        dataset = AV2Dataset(root=f"{path_to_datasets}/{dataset_name}", resize_factor=0.5, subsample_step=step)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Loaded {len(dataset)} samples from {dataset_name} dataset.")

    # Create multi-view anchor dataset
    multiview_dataset = MultiViewAnchorDataset(
        base_dataset=dataset,
        generate_mesh_path=save_path,
        num_views=num_views,
        split_size=split_size,
        split_index=split_index,
        instance_list_path=instance_list_path
    )

    print(f"Loaded {len(multiview_dataset)} multi-view samples ({num_views} views each) from {dataset_name} dataset.")

    return multiview_dataset, dataset
