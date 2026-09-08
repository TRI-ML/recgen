"""
Multi-view anchor dataset for evaluation.

Loads pairs (or N-tuples) of views from the same scene for multi-view evaluation.
Uses instance_list.txt to define view pairs.
"""

import os
import json
import numpy as np

from .base_dataset import BaseDataset


class MultiViewAnchorDataset:
    """
    Dataset that loads multiple views of the same object instance for multi-view evaluation.

    Uses instance_list.txt to define view pairs. Each pair contains:
    - Two (or more) frames from the same scene
    - Same object ID
    - Different viewpoints (frame gap for baseline)

    Returns the first view's pose for evaluation (matches training behavior).
    """

    def __init__(
        self,
        base_dataset: BaseDataset,
        generate_mesh_path: str,
        num_views: int = 2,
        split_size: int = -1,
        split_index: int = -1,
        instance_list_path: str = None
    ):
        """
        Args:
            base_dataset: Underlying BOP dataset (e.g., HBDataset)
            generate_mesh_path: Path to save generated meshes
            num_views: Number of views per sample (default 2)
            split_size: Total number of splits for parallel evaluation
            split_index: Index of this split
            instance_list_path: Custom path to instance_list.txt (optional)
        """
        self.base_dataset = base_dataset
        self.generate_mesh_path = generate_mesh_path
        self.num_views = num_views
        self.pairs = {}

        # Read instance list (view pairs)
        if instance_list_path is not None:
            self.split_path = instance_list_path
        else:
            self.split_path = os.path.join(self.base_dataset.root, 'fixed_split', 'instance_list.txt')
        self.read_instance_list()

        if split_size > 0 and split_index >= 0:
            assert split_index < split_size, "split_index must be less than split_size"
            self.pairs = self.split_dataset(split_size, split_index)

    def read_instance_list(self):
        """
        Read instance_list.txt and parse view pairs.

        Format: test, <seq1> <img1>, <seq2> <img2>, <obj_id>
        """
        if not os.path.exists(self.split_path):
            raise FileNotFoundError(
                f"Instance list not found: {self.split_path}\n"
                f"Run scripts/generate_hb_instance_list.py first to create it."
            )

        with open(self.split_path, 'r') as f:
            lines = f.readlines()
            instance_list = [line.strip() for line in lines if line.strip()]

        # Check if sequences are numeric
        sequences = self.base_dataset.sequences
        numeric_sequences = all(seq.isdigit() for seq in sequences)

        # Track statistics for debugging
        total_lines = len(instance_list)
        obj_not_found = 0
        seq_not_found = 0
        img_not_found = 0
        pairs_found = 0

        for line in instance_list:
            parts = line.split(', ')
            if len(parts) < 4:
                continue

            # Parse view 1
            view1_parts = parts[1].split(' ')
            # Parse view 2
            view2_parts = parts[2].split(' ')

            if numeric_sequences:
                obj_id = int(parts[3].split(" ")[0])
                seq1 = view1_parts[0].zfill(6)
                seq2 = view2_parts[0].zfill(6)
            else:
                obj_id = parts[3].split(" ")[0]
                seq1 = [seq for seq in sequences if view1_parts[0] in seq][0]
                seq2 = [seq for seq in sequences if view2_parts[0] in seq][0]

            img1_id = int(view1_parts[1])
            img2_id = int(view2_parts[1])

            # Find image indices in evaluation_data
            if obj_id not in self.base_dataset.evaluation_data:
                obj_not_found += 1
                continue
            if seq1 not in self.base_dataset.evaluation_data[obj_id]:
                seq_not_found += 1
                continue
            if seq2 not in self.base_dataset.evaluation_data[obj_id]:
                seq_not_found += 1
                continue

            # Find image index by matching image_id
            seq1_images = self.base_dataset.evaluation_data[obj_id][seq1]
            seq2_images = self.base_dataset.evaluation_data[obj_id][seq2]

            img1_indices = [i for i, e in enumerate(seq1_images) if e["image_id"] == img1_id]
            img2_indices = [i for i, e in enumerate(seq2_images) if e["image_id"] == img2_id]

            if not img1_indices or not img2_indices:
                img_not_found += 1
                continue

            img1_index = img1_indices[0]
            img2_index = img2_indices[0]

            if obj_id not in self.pairs:
                self.pairs[obj_id] = []

            # Assertion: View frames must be different (multi-view means different viewpoints)
            assert img1_id != img2_id, (
                f"Multi-view pair must have different frames! "
                f"obj_id={obj_id}, seq1={seq1}, seq2={seq2}, img1={img1_id}, img2={img2_id}"
            )

            self.pairs[obj_id].append({
                'views': [
                    {'seq': seq1, 'image_index': img1_index, 'image_id': img1_id},
                    {'seq': seq2, 'image_index': img2_index, 'image_id': img2_id},
                ],
                'obj_id': obj_id,
            })
            pairs_found += 1

        # Print debug info
        print(f"  [MultiViewAnchorDataset] instance_list parsing:")
        print(f"    Total lines: {total_lines}")
        print(f"    Pairs found: {pairs_found}")
        print(f"    Skipped (obj not in evaluation_data): {obj_not_found}")
        print(f"    Skipped (seq not in evaluation_data): {seq_not_found}")
        print(f"    Skipped (img not found in seq): {img_not_found}")

    def split_dataset(self, split_size: int, split_index: int):
        """Split dataset for parallel evaluation.

        Distributes remainder evenly across the first workers so the max
        difference between any two workers is 1 sample (same logic as
        AnchorDatasetSimple.split_dataset).
        """
        dim = len(self)
        base_size = dim // split_size
        remainder = dim % split_size
        # First 'remainder' workers get (base_size + 1), rest get base_size
        if split_index < remainder:
            start_index = split_index * (base_size + 1)
            end_index = start_index + base_size + 1
        else:
            start_index = remainder * (base_size + 1) + (split_index - remainder) * base_size
            end_index = start_index + base_size

        new_pairs = {}
        current_index = 0
        for obj_id in self.pairs:
            for pair in self.pairs[obj_id]:
                if start_index <= current_index < end_index:
                    if obj_id not in new_pairs:
                        new_pairs[obj_id] = []
                    new_pairs[obj_id].append(pair)
                current_index += 1

        return new_pairs

    def __len__(self):
        total = 0
        for obj_id in self.pairs:
            total += len(self.pairs[obj_id])
        return total

    def __iter__(self):
        self.current_obj_ids = list(self.pairs.keys())
        self.current_obj_index = 0
        self.current_pair_index = 0
        return self

    def __next__(self):
        if self.current_obj_index >= len(self.current_obj_ids):
            raise StopIteration

        obj_id = self.current_obj_ids[self.current_obj_index]
        pairs = self.pairs[obj_id]

        if self.current_pair_index >= len(pairs):
            self.current_obj_index += 1
            self.current_pair_index = 0
            return self.__next__()

        pair = pairs[self.current_pair_index]
        self.current_pair_index += 1

        # Load all views
        views = []
        for view_info in pair['views'][:self.num_views]:
            view_data = self.base_dataset.read(
                obj_id,
                view_info['seq'],
                view_info['image_index']
            )
            views.append(view_data)

        # Assertions: Verify multi-view data integrity
        assert len(views) == self.num_views, (
            f"Expected {self.num_views} views, got {len(views)} for obj_id={obj_id}"
        )

        # All views must have the same object_id
        view_obj_ids = [v['object_id'] for v in views]
        assert all(oid == obj_id for oid in view_obj_ids), (
            f"Object ID mismatch in multi-view pair! Expected obj_id={obj_id}, got {view_obj_ids}"
        )

        # Views must have different image_ids (different viewpoints)
        view_image_ids = [v['image_id'] for v in views]
        assert len(set(view_image_ids)) == len(view_image_ids), (
            f"Duplicate image_ids in multi-view pair! obj_id={obj_id}, image_ids={view_image_ids}"
        )

        # Required keys must be present
        required_keys = ['rgb', 'depth', 'mask', 'camera_intrinsics', 'pose', 'object_id', 'image_id']
        for i, view in enumerate(views):
            for key in required_keys:
                assert key in view, (
                    f"View {i} missing required key '{key}' for obj_id={obj_id}"
                )

        # Create save folder based on first view
        first_view = pair['views'][0]
        obj_folder = os.path.join(
            self.generate_mesh_path,
            f"obj_{str(obj_id).zfill(6)}_multiview_{first_view['seq']}_imgs_" +
            "_".join([str(v['image_id']).zfill(6) for v in pair['views'][:self.num_views]])
        )

        return views, obj_folder

    def read_model(self, object_id: int):
        """Read object model (delegate to base dataset)."""
        return self.base_dataset.read_model(object_id)

    def read_all_models(self):
        """Read all object models (delegate to base dataset)."""
        return self.base_dataset.read_all_models()
