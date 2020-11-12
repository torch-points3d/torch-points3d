import os
import os.path as osp
import torch
import logging
import multiprocessing
from typing import Dict, List, Tuple
from collections import defaultdict
import random

from torch_geometric.data import InMemoryDataset, Data
from torch_points3d.datasets.object_detection.scannet import ScannetObjectDetection
from torch_points3d.datasets.segmentation.scannet import Scannet
import torch_points3d.core.data_transform as cT

log = logging.getLogger(__name__)


class SimpleDataset(InMemoryDataset):
    def __init__(self, data, slices):
        super().__init__()
        self.data = data
        self.slices = slices


class ScannetOneShotDetection(ScannetObjectDetection):
    def __init__(
        self, root, split="train", transform=None, pre_transform=None, pre_filter=None, process_workers=4,
    ):
        super().__init__(
            root,
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            process_workers=process_workers,
            use_instance_bboxes=True,
        )
        if split == "train":
            path = self.processed_paths[3]
        elif split == "val":
            path = self.processed_paths[4]
        elif split == "test":
            path = self.processed_paths[5]
        instances = torch.load(path)
        self.instances = {}
        for label, instance_raw in instances.items():
            self.instances[label] = SimpleDataset(instance_raw[0], instance_raw[1])

    def download(self):
        super().download()

    @property
    def processed_file_names(self):
        file_names = super().processed_file_names
        for s in Scannet.SPLITS:
            if s == "test":
                continue
            file_names.append(s + "_instances.pt")
        return file_names

    @staticmethod
    def extract_instances(data) -> Dict[int, List[Data]]:
        unique_instance_labels = torch.unique(data.instance_labels)
        instances = defaultdict(list)
        for instance_id in unique_instance_labels:
            if instance_id == 0:
                continue
            instance_mask = data.instance_labels == instance_id
            instance_filter = cT.Select(instance_mask)
            instance_data = instance_filter(data)
            label = instance_data.y[0].item()
            if label in Scannet.VALID_CLASS_IDS:
                instances[label].append(instance_data)
        return instances

    def process(self):
        super().process()
        for i, split in enumerate(self.SPLITS):
            if split == "test":
                continue
            data, slices = torch.load(self.processed_paths[i])
            data_wrapper = SimpleDataset(data, slices)
            all_instances = []
            for d in data_wrapper:
                ins = self.extract_instances(d)
                all_instances.append(ins)
            del data_wrapper, data, slices

            instances = defaultdict(list)
            for d in all_instances:
                for k, ins in d.items():
                    instances[k] += ins

            collated_instances = {}
            for instance_label, instance_list in instances.items():
                data, slices = self.collate(instance_list)
                delattr(data, "instance_bboxes")
                delattr(data, "instance_labels")
                collated_instances[instance_label] = (data, slices)
            torch.save(collated_instances, self.processed_paths[i + 3])

    def get_random_instance(self, label):
        dataset = self.instances[label]
        nb_items = len(dataset)
        idx = random.randint(0, nb_items - 1)
        return dataset[idx]


if __name__ == "__main__":
    dataset = ScannetOneShotDetection("/home/ChauletN/torch-points3d/data/scannet-oneshot", process_workers=0)
    instance = dataset.get_random_instance(3)
