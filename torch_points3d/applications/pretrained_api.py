import os
import logging
import urllib.request

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset, instantiate_mock_dataset
from torch_points3d.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

log = logging.getLogger(__name__)

DIR = os.path.dirname(os.path.realpath(__file__))
CHECKPOINT_DIR = os.path.join(DIR, "weights")


def download_file(url, out_file):
    if not os.path.exists(out_file):
        urllib.request.urlretrieve(url, out_file)
    else:
        log.warning("WARNING: skipping download of existing file " + out_file)


class PretainedRegistry(object):

    pointnet2_largemsg = {
        "s3dis": {
            "1": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1e1p0csk/pointnet2_largemsg.pt",
            "2": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2i499g2e/pointnet2_largemsg.pt",
            "3": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1gyokj69/pointnet2_largemsg.pt",
            "4": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1ejjs4s2/pointnet2_largemsg.pt",
            "5": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/etxij0j6/pointnet2_largemsg.pt",
            "6": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/8n8t391d/pointnet2_largemsg.pt",
        }
    }

    pointgroup = {
        "scannet": "https://api.wandb.ai/files/nicolas/panoptic/2ta6vfu2/PointGroup.pt"
    }

    @staticmethod
    def from_pretrained(model_name, dataset_name, fold=None, download=True, out_file=None, weight_name='latest', mock_dataset=True):
        # Convert inputs to registry format
        model_name = model_name.lower()
        dataset_name = dataset_name.lower()
        fold = str(fold) if fold is not None else fold

        model = PretainedRegistry.__dict__.get(model_name)
        if model:
            dataset = model.get(dataset_name)
            if dataset:
                if isinstance(dataset, dict):
                    if fold is None:
                        raise Exception(
                            "Fold should be provided. Here are options {}".format(list(dataset.keys())))
                    if fold not in dataset.keys():
                        raise Exception(
                            "Fold {} doesn't exist within the dataset registry. Here are options {}".format(fold, list(dataset.keys())))
                    url = dataset.get(fold)
                else:
                    url = dataset
            else:
                raise Exception("This model {} doesn't have this dataset {}. Only {}".format(
                    dataset, model.keys()))
        else:
            raise Exception(
                "The model_name {} doesn't exist within the registry. List of available models: {}".format(
                    model_name, PretainedRegistry.available_models()))

        if out_file is None:
            checkpoint_model_name = "_".join([model_name, dataset_name])
            if fold is not None:
                checkpoint_model_name += '_{}'.format(fold)
            else:
                checkpoint_model_name = out_file.replace('.pt', '')

            out_file = checkpoint_model_name + '.pt'
            out_file = os.path.join(CHECKPOINT_DIR, out_file)

        if download:
            download_file(url, out_file)

            weight_name = weight_name if weight_name is not None else "latest"

            checkpoint: ModelCheckpoint = ModelCheckpoint(
                CHECKPOINT_DIR,
                checkpoint_model_name,
                weight_name if weight_name is not None else "latest",
                resume=True,
            )
            if mock_dataset:
                dataset = instantiate_mock_dataset(checkpoint.data_config)
            else:
                dataset = instantiate_dataset(checkpoint.data_config)

            import pdb
            pdb.set_trace()

            model: BaseModel = checkpoint.create_model(
                dataset, weight_name=weight_name
            )

            return model, url
        else:
            return None, url

    @ staticmethod
    def available_models():
        excluded = ['from_pretrained', 'available_models']
        return list([m for m in PretainedRegistry.__dict__.keys()
                     if (m not in excluded and '__' not in m)])
