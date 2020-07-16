import os
import logging
import urllib.request

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset
from torch_points3d.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

from torch_points3d.utils.wandb_utils import Wandb
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

log = logging.getLogger(__name__)

DIR = os.path.dirname(os.path.realpath(__file__))
CHECKPOINT_DIR = os.path.join(DIR, "weights")


def download_file(url, out_file):
    if not os.path.exists(out_file):
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        urllib.request.urlretrieve(url, out_file)
    else:
        log.warning("WARNING: skipping download of existing file " + out_file)


class PretainedRegistry(object):

    MODELS = {
        "pointnet2_largemsg-s3dis-1": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1e1p0csk/pointnet2_largemsg.pt",
        "pointnet2_largemsg-s3dis-2": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2i499g2e/pointnet2_largemsg.pt",
        "pointnet2_largemsg-s3dis-3": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1gyokj69/pointnet2_largemsg.pt",
        "pointnet2_largemsg-s3dis-4": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1ejjs4s2/pointnet2_largemsg.pt",
        "pointnet2_largemsg-s3dis-5": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/etxij0j6/pointnet2_largemsg.pt",
        "pointnet2_largemsg-s3dis-6": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/8n8t391d/pointnet2_largemsg.pt",
        "pointgroup-scannet": "https://api.wandb.ai/files/nicolas/panoptic/2ta6vfu2/PointGroup.pt",
    }

    MOCK_USED_PROPERTIES = {
        "pointnet2_largemsg-s3dis-1": {"feature_dimension": 4, "num_classes": 13},
        "pointnet2_largemsg-s3dis-2": {"feature_dimension": 4, "num_classes": 13},
        "pointnet2_largemsg-s3dis-3": {"feature_dimension": 4, "num_classes": 13},
        "pointnet2_largemsg-s3dis-4": {"feature_dimension": 4, "num_classes": 13},
        "pointnet2_largemsg-s3dis-5": {"feature_dimension": 4, "num_classes": 13},
        "pointnet2_largemsg-s3dis-6": {"feature_dimension": 4, "num_classes": 13},
        "pointgroup-scannet": {},
    }

    @staticmethod
    def from_pretrained(model_tag, download=True, out_file=None, weight_name="latest", mock_dataset=True):
        # Convert inputs to registry format

        if PretainedRegistry.MODELS.get(model_tag) is not None:
            url = PretainedRegistry.MODELS.get(model_tag)
        else:
            raise Exception(
                "model_tag {} doesn't exist within available models. Here is the list of pre-trained models {}".format(
                    model_tag, PretainedRegistry.available_models()
                )
            )

        checkpoint_name = model_tag + ".pt"
        out_file = os.path.join(CHECKPOINT_DIR, checkpoint_name)

        if download:
            download_file(url, out_file)

            weight_name = weight_name if weight_name is not None else "latest"

            checkpoint: ModelCheckpoint = ModelCheckpoint(
                CHECKPOINT_DIR, model_tag, weight_name if weight_name is not None else "latest", resume=False,
            )
            if mock_dataset:
                dataset = checkpoint.data_config
                if PretainedRegistry.MOCK_USED_PROPERTIES.get(model_tag) is not None:
                    for k, v in PretainedRegistry.MOCK_USED_PROPERTIES.get(model_tag).items():
                        dataset[k] = v
            else:
                dataset = instantiate_dataset(checkpoint.data_config)

            model: BaseModel = checkpoint.create_model(dataset, weight_name=weight_name)

            Wandb.set_urls_to_model(model, url)

            return model

    @staticmethod
    def available_models():
        return PretainedRegistry.MODELS.keys()
