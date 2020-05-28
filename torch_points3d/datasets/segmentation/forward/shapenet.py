import torch
import glob
import os
from torch_geometric.io import read_txt_array
from torch_geometric.data.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import knn_interpolate
import numpy as np
import logging

from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.utils import is_list
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.shapenet_part_tracker import ShapenetPartTracker
from torch_points3d.datasets.segmentation.shapenet import ShapeNet

log = logging.getLogger(__name__)


class _ForwardShapenet(torch.utils.data.Dataset):
    """ Dataset to run forward inference on Shapenet kind of data data. Runs on a whole folder.
    Arguments:
        path: folder that contains a set of files of a given category
        category: index of the category to use for forward inference. This value depends on how many categories the model has been trained one.
        transforms: transforms to be applied to the data
        include_normals: wether to include normals for the forward inference
    """

    def __init__(self, path, category: int, transforms=None, include_normals=True):
        super().__init__()
        self._category = category
        self._path = path
        self._files = sorted(glob.glob(os.path.join(self._path, "*.txt")))
        self._transforms = transforms
        self._include_normals = include_normals
        assert os.path.exists(self._path)
        if self.__len__() == 0:
            raise ValueError("Empty folder %s" % path)

    def __len__(self):
        return len(self._files)

    def _read_file(self, filename):
        raw = read_txt_array(filename)
        pos = raw[:, :3]
        x = raw[:, 3:6]
        if raw.shape[1] == 7:
            y = raw[:, 6].type(torch.long)
        else:
            y = None
        return Data(pos=pos, x=x, y=y)

    def get_raw(self, index):
        """ returns the untransformed data associated with an element
        """
        return self._read_file(self._files[index])

    @property
    def num_features(self):
        feats = self[0].x
        if feats is not None:
            return feats.shape[-1]
        return 0

    def get_filename(self, index):
        return os.path.basename(self._files[index])

    def __getitem__(self, index):
        data = self._read_file(self._files[index])
        category = torch.ones(data.pos.shape[0], dtype=torch.long) * self._category
        setattr(data, "category", category)
        setattr(data, "sampleid", torch.tensor([index]))
        if not self._include_normals:
            data.x = None
        if self._transforms is not None:
            data = self._transforms(data)
        return data


class ForwardShapenetDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        forward_category = dataset_opt.forward_category
        if not isinstance(forward_category, str):
            raise ValueError(
                "dataset_opt.forward_category is not set or is not a string. Current value: {}".format(
                    dataset_opt.forward_category
                )
            )
        self._train_categories = dataset_opt.category
        if not is_list(self._train_categories):
            self._train_categories = [self._train_categories]

        # Sets the index of the category with respect to the categories in the trained model
        self._cat_idx = None
        for i, train_category in enumerate(self._train_categories):
            if forward_category.lower() == train_category.lower():
                self._cat_idx = i
                break
        if self._cat_idx is None:
            raise ValueError(
                "Cannot run an inference on category {} with a network trained on {}".format(
                    forward_category, self._train_categories
                )
            )
        log.info(
            "Running an inference on category {} with a network trained on {}".format(
                forward_category, self._train_categories
            )
        )

        self._data_path = dataset_opt.dataroot
        include_normals = dataset_opt.include_normals if dataset_opt.include_normals else True

        transforms = SaveOriginalPosId()
        for t in [self.pre_transform, self.test_transform]:
            if t:
                transforms = T.Compose([transforms, t])
        self.test_dataset = _ForwardShapenet(
            self._data_path, self._cat_idx, transforms=transforms, include_normals=include_normals
        )

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return ShapenetPartTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

    def predict_original_samples(self, batch, conv_type, output):
        """ Takes the output generated by the NN and upsamples it to the original data
        Arguments:
            batch -- processed batch
            conv_type -- Type of convolutio (DENSE, PARTIAL_DENSE, etc...)
            output -- output predicted by the model
        """
        full_res_results = {}
        num_sample = BaseDataset.get_num_samples(batch, conv_type)
        if conv_type == "DENSE":
            output = output.reshape(num_sample, -1, output.shape[-1])  # [B,N,L]

        setattr(batch, "_pred", output)
        for b in range(num_sample):
            sampleid = batch.sampleid[b]
            sample_raw_pos = self.test_dataset[0].get_raw(sampleid).pos.to(output.device)
            predicted = BaseDataset.get_sample(batch, "_pred", b, conv_type)
            origindid = BaseDataset.get_sample(batch, SaveOriginalPosId.KEY, b, conv_type)
            full_prediction = knn_interpolate(predicted, sample_raw_pos[origindid], sample_raw_pos, k=3)
            labels = full_prediction.max(1)[1].unsqueeze(-1)
            full_res_results[self.test_dataset[0].get_filename(sampleid)] = np.hstack(
                (sample_raw_pos.cpu().numpy(), labels.cpu().numpy(),)
            )
        return full_res_results

    @property
    def class_to_segments(self):
        classes_to_segment = {}
        for key in self._train_categories:
            classes_to_segment[key] = ShapeNet.seg_classes[key]
        return classes_to_segment

    @property
    def num_classes(self):
        segments = self.class_to_segments.values()
        num = 0
        for seg in segments:
            num = max(num, max(seg))
        return num + 1
