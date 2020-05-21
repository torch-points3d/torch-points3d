import os
import sys
import logging

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(ROOT))

log = logging.getLogger(__name__)


class ModelInference(object):
    """ Base class transform for performing a point cloud inference using a pre_trained model
    Subclass and implement the ``__call__`` method with your own forward. 
    See ``PointNetForward`` for an example implementation.
    
    Parameters
    ----------
    checkpoint_dir: str
        Path to a checkpoint directory
    model_name: str
        Model name, the file ``checkpoint_dir/model_name.pt`` must exist
    """

    def __init__(self, checkpoint_dir, model_name, weight_name, feat_name, num_classes=None, mock_dataset=True):
        # Checkpoint
        from torch_points3d.datasets.base_dataset import BaseDataset
        from torch_points3d.datasets.dataset_factory import instantiate_dataset
        from torch_points3d.utils.mock import MockDataset
        import torch_points3d.metrics.model_checkpoint as model_checkpoint

        checkpoint = model_checkpoint.ModelCheckpoint(checkpoint_dir, model_name, weight_name, strict=True)
        if mock_dataset:
            dataset = MockDataset(num_classes)
            dataset.num_classes = num_classes
        else:
            dataset = instantiate_dataset(checkpoint.data_config)
        BaseDataset.set_transform(self, checkpoint.data_config)
        self.model = checkpoint.create_model(dataset, weight_name=weight_name)
        self.model.eval()

    def __call__(self, data):
        raise NotImplementedError


class PointNetForward(ModelInference):
    """ Transform for running a PointNet inference on a Data object. It assumes that the
    model has been trained for segmentation.
    
    Parameters
    ----------
    checkpoint_dir: str
        Path to a checkpoint directory
    model_name: str
        Model name, the file ``checkpoint_dir/model_name.pt`` must exist
    weight_name: str
        Type of weights to load (best for iou, best for loss etc...)
    feat_name: str
        Name of the key in Data that will hold the output of the forward
    num_classes: int
        Number of classes that the model was trained on
    """

    def __init__(self, checkpoint_dir, model_name, weight_name, feat_name, num_classes, mock_dataset=True):
        super(PointNetForward, self).__init__(
            checkpoint_dir, model_name, weight_name, feat_name, num_classes=num_classes, mock_dataset=mock_dataset
        )
        self.feat_name = feat_name

        from torch_points3d.datasets.base_dataset import BaseDataset
        from torch_geometric.transforms import FixedPoints, GridSampling3D

        self.inference_transform = BaseDataset.remove_transform(self.inference_transform, [GridSampling3D, FixedPoints])

    def __call__(self, data):
        data_c = data.clone()
        data_c.pos = data_c.pos.float()
        if self.inference_transform:
            data_c = self.inference_transform(data_c)
        self.model.set_input(data_c, data.pos.device)
        feat = self.model.get_local_feat().detach()
        setattr(data, str(self.feat_name), feat)
        return data

    def __repr__(self):
        return "{}(model: {}, transform: {})".format(
            self.__class__.__name__, self.model.__class__.__name__, self.inference_transform
        )
