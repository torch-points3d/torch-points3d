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
        "minkowski-res16-s3dis-1": "https://api.wandb.ai/files/nicolas/s3dis-benchmark/1fyr7ri9/Res16UNet34C.pt",
        "minkowski-res16-s3dis-2": "https://api.wandb.ai/files/nicolas/s3dis-benchmark/1gdgx2ni/Res16UNet34C.pt",
        "minkowski-res16-s3dis-3": "https://api.wandb.ai/files/nicolas/s3dis-benchmark/gt3ttamp/Res16UNet34C.pt",
        "minkowski-res16-s3dis-4": "https://api.wandb.ai/files/nicolas/s3dis-benchmark/36yxu3yc/Res16UNet34C.pt",
        "minkowski-res16-s3dis-5": "https://api.wandb.ai/files/nicolas/s3dis-benchmark/2r0tsub1/Res16UNet34C.pt",
        "minkowski-res16-s3dis-6": "https://api.wandb.ai/files/nicolas/s3dis-benchmark/30yrkk5p/Res16UNet34C.pt",
        "minkowski-registration-3dmatch": "https://api.wandb.ai/files/humanpose1/registration/2wvwf92e/MinkUNet_Fragment.pt",
        "minkowski-registration-kitti": "https://api.wandb.ai/files/humanpose1/KITTI/2xpy7u1i/MinkUNet_Fragment.pt",
        "minkowski-registration-modelnet": "https://api.wandb.ai/files/humanpose1/modelnet/39u5v3bm/MinkUNet_Fragment.pt",
        "rsconv-s3dis-1": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2b99o12e/RSConv_MSN_S3DIS.pt",
        "rsconv-s3dis-2": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1onl4h59/RSConv_MSN_S3DIS.pt",
        "rsconv-s3dis-3": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2cau6jua/RSConv_MSN_S3DIS.pt",
        "rsconv-s3dis-4": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1qqmzgnz/RSConv_MSN_S3DIS.pt",
        "rsconv-s3dis-5": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/378enxsu/RSConv_MSN_S3DIS.pt",
        "rsconv-s3dis-6": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/23f4upgc/RSConv_MSN_S3DIS.pt",
        "kpconv-s3dis-1": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/okiba8gp/KPConvPaper.pt",
        "kpconv-s3dis-2": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2at56wrm/KPConvPaper.pt",
        "kpconv-s3dis-3": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1ipv9lso/KPConvPaper.pt",
        "kpconv-s3dis-4": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2c13jhi0/KPConvPaper.pt",
        "kpconv-s3dis-5": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1kf8yg5s/KPConvPaper.pt",
        "kpconv-s3dis-6": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2ph7ejss/KPConvPaper.pt",
    }

    MOCK_USED_PROPERTIES = {
        "pointnet2_largemsg-s3dis-1": {"feature_dimension": 4, "num_classes": 13},
        "pointnet2_largemsg-s3dis-2": {"feature_dimension": 4, "num_classes": 13},
        "pointnet2_largemsg-s3dis-3": {"feature_dimension": 4, "num_classes": 13},
        "pointnet2_largemsg-s3dis-4": {"feature_dimension": 4, "num_classes": 13},
        "pointnet2_largemsg-s3dis-5": {"feature_dimension": 4, "num_classes": 13},
        "pointnet2_largemsg-s3dis-6": {"feature_dimension": 4, "num_classes": 13},
        "pointgroup-scannet": {},
        "rsconv-s3dis-1": {"feature_dimension": 4, "num_classes": 13},
        "rsconv-s3dis-2": {"feature_dimension": 4, "num_classes": 13},
        "rsconv-s3dis-3": {"feature_dimension": 4, "num_classes": 13},
        "rsconv-s3dis-4": {"feature_dimension": 4, "num_classes": 13},
        "rsconv-s3dis-5": {"feature_dimension": 4, "num_classes": 13},
        "rsconv-s3dis-6": {"feature_dimension": 4, "num_classes": 13},
        "minkowski-res16-s3dis-1": {"feature_dimension": 4, "num_classes": 13},
        "minkowski-res16-s3dis-2": {"feature_dimension": 4, "num_classes": 13},
        "minkowski-res16-s3dis-3": {"feature_dimension": 4, "num_classes": 13},
        "minkowski-res16-s3dis-4": {"feature_dimension": 4, "num_classes": 13},
        "minkowski-res16-s3dis-5": {"feature_dimension": 4, "num_classes": 13},
        "minkowski-res16-s3dis-6": {"feature_dimension": 4, "num_classes": 13},
        "minkowski-registration-3dmatch": {"feature_dimension": 1},
        "minkowski-registration-kitti": {"feature_dimension": 1},
        "minkowski-registration-modelnet": {"feature_dimension": 1},
        "kpconv-s3dis-1": {"feature_dimension": 4, "num_classes": 13},
        "kpconv-s3dis-2": {"feature_dimension": 4, "num_classes": 13},
        "kpconv-s3dis-3": {"feature_dimension": 4, "num_classes": 13},
        "kpconv-s3dis-4": {"feature_dimension": 4, "num_classes": 13},
        "kpconv-s3dis-5": {"feature_dimension": 4, "num_classes": 13},
        "kpconv-s3dis-6": {"feature_dimension": 4, "num_classes": 13},
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

            BaseDataset.set_transform(model, checkpoint.data_config)

            return model

    @staticmethod
    def available_models():
        return PretainedRegistry.MODELS.keys()
