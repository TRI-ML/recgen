import copy
from typing import List
import json

import os

import numpy as np
import trimesh
import cv2


from .base_dataset import BaseDataset
from ..core.utils import symmetry_tfs_from_info


class BOPDataset(BaseDataset):
    def __init__(self, root: str, dataset_prefix: str, test_dir: str = "test_all", sequences: List = [],
                 test_subdir: str = "test", camera_file: str = "camera.json",
                 resize_factor: float = 1.0,
                 filter_with_test_file: bool = True,
                 subsample_step: int = 1):
        super().__init__(root=root, dataset_prefix=dataset_prefix, test_dir=test_dir, test_subdir=test_subdir)

        self.resize_factor = resize_factor

        self.mesh_dir = os.path.join(self.root, self.dataset_prefix + "_models", "models_eval")
        self.base_dir = os.path.join(self.root, self.dataset_prefix + "_base", self.dataset_prefix)
        self.sequences_dir = os.path.join(self.root, self.dataset_prefix + "_" + self.test_dir, self.test_subdir)
        self.camera_dir = os.path.join(self.base_dir, camera_file)
        # load camera infos
        with open(self.camera_dir, 'r') as f:
            self.camera_info = json.load(f)
        self.width = int(self.camera_info['width'] * self.resize_factor)
        self.height = int(self.camera_info['height'] * self.resize_factor)

        if len(sequences) == 0:
            # load all sequences
            self.sequences = [f for f in os.listdir(self.sequences_dir) if os.path.isdir(os.path.join(self.sequences_dir, f))]
            self.sequences.sort()
        else:
            self.sequences = [str(seq).zfill(6) for seq in sequences]

        self.filter_with_test_file = filter_with_test_file
        if self.filter_with_test_file:
            self.load_test_targets()
        self.load_data()

        self.subsample_step = subsample_step
        if subsample_step > 1:
            for obj_id in self.evaluation_data:
                for seq in self.evaluation_data[obj_id]:
                    self.evaluation_data[obj_id][seq] = self.evaluation_data[obj_id][seq][::subsample_step]

    def load_test_targets(self):
        targets_path = os.path.join(self.base_dir, "test_targets_bop19.json")
        if os.path.exists(targets_path):
            with open(targets_path, 'r') as f:
                targets = json.load(f)
            # reorganize targets
            targets_org = {}
            for elem in targets:
                seq = str(elem['scene_id']).zfill(6)
                if seq not in targets_org:
                    targets_org[seq] = {}
                if str(elem['im_id']) not in targets_org[seq]:
                    targets_org[seq][str(elem['im_id'])] = []
                targets_org[seq][str(elem['im_id'])].append(elem['obj_id'])
            self.test_targets = targets_org


    def load_data(self):

        filter_data = False
        # if the targets are available, filter the data to only load those objects
        if len(self.test_targets) > 0:
            filter_data = True

        for seq in self.sequences:
            if filter_data:
                if seq not in self.test_targets:
                    continue

            images = [f for f in os.listdir(os.path.join(self.sequences_dir, seq, "rgb")) if f.endswith(".png")]
            images.sort()
            images_dict = {str(int(images[i].split(".")[0])): images[i] for i in range(len(images))}
            self.sequence_to_images[seq] = images

            gt_path = os.path.join(self.sequences_dir, seq, "scene_gt.json")
            with open(gt_path, 'r') as f:
                gts = json.load(f)


            gt_info_path = os.path.join(self.sequences_dir, seq, "scene_gt_info.json")
            if os.path.exists(gt_info_path):
                with open(gt_info_path, 'r') as f:
                    gt_info = json.load(f)
                for k in gts.keys():
                    for i in range(len(gts[k])):
                        gts[k][i]['visibility'] = gt_info[k][i]['visib_fract']
            else:
                for k in gts.keys():
                    for i in range(len(gts[k])):
                        gts[k][i]['visibility'] = 1.0
            self.sequence_to_gts[seq] = gts

            # attach gt info

            cam_path = os.path.join(self.sequences_dir, seq, "scene_camera.json")
            with open(cam_path, 'r') as f:
                cams = json.load(f)
            self.sequence_to_cam[seq] = cams

            for k, gt in gts.items():
                if filter_data:
                    if k not in self.test_targets[seq]:
                        continue

                image_name = images_dict[k]
                cam = cams[k]

                for i, obj in enumerate(gt):

                    if filter_data:
                        if obj['obj_id'] not in self.test_targets[seq][k]:
                            print(f"Skipping object {obj['obj_id']} in sequence {seq} image {k}")
                            continue
                        if obj['visibility'] < 0.25:
                            print(f"Skipping object {obj['obj_id']} in sequence {seq} image {k} due to low visibility {obj['visibility']}")
                            continue

                    obj_id = obj['obj_id']
                    obj_R = np.array(obj['cam_R_m2c']).reshape(3, 3)
                    obj_t = np.array(obj['cam_t_m2c']).reshape(3, 1)

                    if obj_id not in self.evaluation_data:
                        self.evaluation_data[obj_id] = {}
                    if seq not in self.evaluation_data[obj_id]:
                        self.evaluation_data[obj_id][seq] = []

                    mask_name = image_name.split(".")[0] + "_" + str(i).zfill(6) + ".png"

                    evaluation_elem = {
                        'image_id': int(k),
                        'R': obj_R,
                        't': obj_t,
                        'file_name': image_name,
                        'mask_name': mask_name,
                        'cam_K': np.array(cam['cam_K']).reshape(3, 3),
                        'depth_scale': cam['depth_scale'],
                    }
                    self.evaluation_data[obj_id][seq].append(evaluation_elem)

    def __len__(self):
        length = 0
        for object_id in self.evaluation_data:
            for sequence in self.evaluation_data[object_id]:
                length += len(self.evaluation_data[object_id][sequence])
        return length

    def __iter__(self):
        self.current_object_idx = 0
        self.current_sequence_idx = 0
        self.current_image_idx = 0
        return self

    def __next__(self):
        if self.current_object_idx >= len(self.evaluation_data):
            raise StopIteration

        object_ids = list(self.evaluation_data.keys())
        current_object_id = object_ids[self.current_object_idx]
        sequences = list(self.evaluation_data[current_object_id].keys())

        if self.current_sequence_idx >= len(sequences):
            self.current_object_idx += 1
            self.current_sequence_idx = 0
            self.current_image_idx = 0
            return self.__next__()

        current_sequence = sequences[self.current_sequence_idx]
        images = self.evaluation_data[current_object_id][current_sequence]

        if self.current_image_idx >= len(images):
            self.current_sequence_idx += 1
            self.current_image_idx = 0
            return self.__next__()

        current_image = self.current_image_idx
        data = self.read(current_object_id, current_sequence, current_image)

        self.current_image_idx += 1

        return data


    def read(self, object_id: int, sequence: str, image: int):

        element = self.evaluation_data[object_id][sequence][image]
        file_name = element['file_name']
        rgb_path = os.path.join(self.sequences_dir, sequence, "rgb", file_name)
        depth_path = os.path.join(self.sequences_dir, sequence, "depth", file_name)
        mask_path = os.path.join(self.sequences_dir, sequence, "mask_visib", element['mask_name'])

        camera_intrinsics = copy.deepcopy(element['cam_K'])
        depth_scale = element['depth_scale']
        obj_R = element['R']
        obj_t = element['t'] * 0.001  # convert to meters
        pose = np.eye(4)
        pose[:3, :3] = obj_R
        pose[:3, 3] = obj_t.flatten()

        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # convert to meters
        depth_image /= depth_scale
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if self.resize_factor != 1.0:
            new_width = int(self.width)
            new_height = int(self.height)
            rgb_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            depth_image = cv2.resize(depth_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            camera_intrinsics[0, 0] *= self.resize_factor
            camera_intrinsics[1, 1] *= self.resize_factor
            camera_intrinsics[0, 2] *= self.resize_factor
            camera_intrinsics[1, 2] *= self.resize_factor


        dataset_element = {
            'rgb': rgb_image,
            'depth': depth_image,
            'camera_intrinsics': copy.deepcopy(camera_intrinsics),
            'mask': mask,
            'R': obj_R,
            't': obj_t,
            'pose': pose,
            'sequence': sequence,
            'file_name': file_name,
            'object_id': object_id,
            'image_id': element['image_id'],
        }
        return dataset_element

    def read_model(self, object_id: int):
        model_path = os.path.join(self.mesh_dir, f"obj_{str(object_id).zfill(6)}.ply")
        mesh = trimesh.load(model_path)
        mesh.vertices *= 1e-3
        metadata = {}

        # read the symmetries if exist
        model_info = os.path.join(self.mesh_dir, "models_info.json")
        with open(model_info, 'r') as f:
            info = json.load(f)
        if str(object_id) in info:
            metadata['diameter'] = info[str(object_id)]['diameter'] * 0.001  # convert to meters
            metadata['min_x'] = info[str(object_id)]['min_x'] * 0.001
            metadata['size_x'] = info[str(object_id)]['size_x'] * 0.001
            metadata['min_y'] = info[str(object_id)]['min_y'] * 0.001
            metadata['size_y'] = info[str(object_id)]['size_y'] * 0.001
            metadata['min_z'] = info[str(object_id)]['min_z'] * 0.001
            metadata['size_z'] = info[str(object_id)]['size_z'] * 0.001

            sym_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]
            symmetry_tfs = None
            if 'symmetries_discrete' in info[str(object_id)]:
                symmetry_tfs = symmetry_tfs_from_info(info[str(object_id)], rot_angle_discrete=5)
                symmetries = info[str(object_id)]['symmetries_discrete']
                for sym in symmetries:
                    rt_mat = np.array(sym).reshape(4, 4)
                    R = rt_mat[:3, :3]
                    t = rt_mat[:3, 3] * 0.001
                    t = t.reshape(3, 1)
                    sym_disc.append({"R": R, "t": t})

            metadata['symmetries_discrete'] = sym_disc
            metadata['symmetry_tfs'] = symmetry_tfs

        return mesh, metadata

    def read_all_models(self):
        models = {}
        model_files = [f for f in os.listdir(self.mesh_dir) if f.endswith(".ply")]
        model_files.sort()
        for model_file in model_files:
            object_id = int(model_file.split("_")[1].split(".")[0])
            mesh, metadata = self.read_model(object_id)
            models[object_id] = (mesh, metadata)
        return models


class HBDataset(BOPDataset):
    def __init__(self, root: str = "", test_dir: str = "val_kinect", test_subdir: str = "val_kinect", resize_factor: float = 1.0, subsample_step: int = 1, load_all_frames: bool = False):
        """
        Args:
            test_subdir: Subdirectory within hb_{test_dir}/ (default: "val_kinect").
                         Use "val_kinect_corrected_5" or "val_kinect_corrected_8" for MoGe-filtered depth.
            load_all_frames: If True, load all frames instead of subsampling by ratio=40.
                            Useful for multi-view evaluation where second views may be at any frame.
        """
        self.load_all_frames = load_all_frames
        super().__init__(root=root, dataset_prefix="hb", test_dir=test_dir, test_subdir=test_subdir, camera_file="camera_kinect.json", resize_factor=resize_factor, subsample_step=subsample_step)

    # override
    def load_test_targets(self):

        # iterate through all the sequences and load the gts
        targets_org = {}
        for seq in self.sequences:

            targets_org[seq] = {}

            gt_path = os.path.join(self.sequences_dir, seq, "scene_gt.json")
            with open(gt_path, 'r') as f:
                gts = json.load(f)


            get_info_path = os.path.join(self.sequences_dir, seq, "scene_gt_info.json")
            with open(get_info_path, 'r') as f:
                gts_info = json.load(f)

            # iterate through frames - either all or every 40th
            ratio = 40
            for k, info in gts_info.items():
                image_id = int(k)
                # Skip frame if not load_all_frames and not divisible by ratio
                if not self.load_all_frames and image_id % ratio != 0:
                    continue

                if str(image_id) not in targets_org[seq]:
                    targets_org[seq][str(image_id)] = []

                frame_gts = gts[str(image_id)]

                for index, obj in enumerate(info):
                    obj_id = frame_gts[index]['obj_id']

                    if obj['visib_fract'] < 0.4:
                        continue

                    if obj_id not in targets_org[seq][str(image_id)]:
                        targets_org[seq][str(image_id)].append(obj_id)

        self.test_targets = targets_org


class AV2Dataset(BOPDataset):
    def __init__(self, root: str = "", test_dir: str = "test_all", resize_factor: float = 1.0, subsample_step: int = 1):
        super().__init__(root=root, dataset_prefix="artvip", test_dir=test_dir, resize_factor=resize_factor,
                         filter_with_test_file=True, subsample_step=subsample_step)

    def load_test_targets(self):
        targets_path = os.path.join(self.base_dir, "test_targets_all.json")
        if os.path.exists(targets_path):
            with open(targets_path, 'r') as f:
                targets = json.load(f)
            targets_org = {}
            for elem in targets:
                seq = str(elem['scene_id']).zfill(6)
                if seq not in targets_org:
                    targets_org[seq] = {}
                if str(elem['im_id']) not in targets_org[seq]:
                    targets_org[seq][str(elem['im_id'])] = []
                targets_org[seq][str(elem['im_id'])].append(elem['obj_id'])
            self.test_targets = targets_org
