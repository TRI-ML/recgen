from abc import ABC, abstractmethod

import numpy as np

class BaseDataset(ABC):
    def __init__(self, root: str, dataset_prefix: str, test_dir: str = "test_all", test_subdir: str = "test"):
        self.root = root

        self.dataset_prefix = dataset_prefix
        self.test_dir = test_dir
        self.test_subdir = test_subdir

        self.width = None
        self.height = None
        self.sequences = None

        # create dictionary with image name for each sequence
        self.objects = []
        self.sequence_to_images = {}
        self.sequence_to_gts = {}
        self.sequence_to_gts_infos = {}
        self.sequence_to_cam = {}
        self.evaluation_data = {}
        self.test_targets = {}

        self.mesh_dir = None
        self.base_dir = None
        self.sequences_dir = None
        self.camera_dir = None

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def read(self, object_id: int, sequence: str, image: int):
        pass

    @abstractmethod
    def read_model(self, object_id: int):
        pass
