import logging
import src.metrics.model_checkpoint as model_checkpoint
from src.core.data_transform.transforms import GridSampling
from test.mockdatasets import MockDataset

log = logging.getLogger(__name__)

class ModelInference(object):
    r"""This transform is responsible to perform a point cloud inference using a pre_trained model
    Args:
        checkpoint_dir (str): Path to a checkpoint_dir
        model_name (str): Model name
    """
    def __init__(self, checkpoint_dir, model_name, weight_name, feat_name, num_classes=None, mock_dataset=True):
        # Checkpoint
        from src.datasets.base_dataset import BaseDataset
        from src.datasets.dataset_factory import instantiate_dataset
        checkpoint = model_checkpoint.ModelCheckpoint(checkpoint_dir, 
                                     model_name, 
                                     weight_name, 
                                     strict=True)
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

    def __init__(self, checkpoint_dir, model_name, weight_name, feat_name, num_classes, mock_dataset=True):
        super(PointNetForward, self).__init__(checkpoint_dir, model_name, weight_name, feat_name, num_classes=num_classes, mock_dataset=mock_dataset)
        self.feat_name = feat_name
        
        from src.datasets.base_dataset import BaseDataset
        from torch_geometric.transforms import FixedPoints
        self.inference_transform = BaseDataset.remove_transform(self.inference_transform, [GridSampling, FixedPoints])

    def __call__(self, data):
        data_c = data.clone()
        data_c.pos = data_c.pos.float()
        if self.inference_transform:
            data_c = self.inference_transform(data_c)
        self.model.set_input(data_c)
        feat = self.model.get_local_feat().detach()
        setattr(data, str(self.feat_name), feat)
        return data

    def __repr__(self):
        return "{}(model: {}, transform: {})".format(self.__class__.__name__, self.model.__class__.__name__, self.inference_transform)