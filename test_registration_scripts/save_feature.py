import torch
import hydra
import logging
from omegaconf import OmegaConf
import os
import os.path as osp
import sys
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

# Import building function for model and dataset
from src.datasets.dataset_factory import instantiate_dataset, get_dataset_class
from src.models.model_factory import instantiate_model

from src.models.base_model import BaseModel
from src.datasets.base_dataset import BaseDataset

# Import from metrics
from src.metrics.colored_tqdm import Coloredtqdm as Ctq
from src.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from src.utils.colors import COLORS

log = logging.getLogger(__name__)


def save(out_path, scene_name, pc_name, data, feature):
    """
    save pointcloud, feature and keypoint if it is asked
    """

    kp = None
    if len(feature) != len(data.pos):
        # it must contain keypoints
        assert getattr(data, "keypoints", None) is not None
        kp = data["keypoints"]
    else:
        kp = np.arange(0, len(feature))
    out_dir = osp.join(out_path, scene_name)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    out_file = osp.join(out_dir, pc_name.split(".")[0] + "_desc.npz")
    np.savez(out_file, pcd=data.pos.numpy(), feat=feature, keypoints=kp)


def run(model: BaseModel, dataset: BaseDataset, device, output_path, cfg):
    # Set dataloaders
    num_fragment = dataset.num_fragment
    if cfg.data.is_patch:
        for i in range(num_fragment):
            dataset.set_patches(i)
            dataset.create_dataloaders(
                model, cfg.batch_size, False, cfg.num_workers, False,
            )
            loader = dataset.test_dataloaders()[0]
            features = []
            scene_name, pc_name = dataset.get_name(i)

            with Ctq(loader) as tq_test_loader:
                for data in tq_test_loader:
                    data = data.to(device)
                    with torch.no_grad():
                        model.set_input(data)
                        model.forward()
                        features.append(model.get_output().cpu())
            features = torch.cat(features, 0).numpy()
            log.info("save {} from {} in  {}".format(pc_name, scene_name, output_path))
            save(output_path, scene_name, pc_name, dataset.base_dataset[i].to("cpu"), features)
    else:
        dataset.create_dataloaders(
            model, 1, False, cfg.num_workers, False,
        )
        loader = dataset.test_dataloaders()[0]
        with Ctq(loader) as tq_test_loader:
            for i, data in enumerate(tq_test_loader):
                data = data.to(device)
                with torch.no_grad():
                    model.set_input(data)
                    model.forward()
                    features = model.get_output()[0]  # batch of 1
                    save(output_path, scene_name, pc_name, data.to("cpu"), features)


@hydra.main(config_path="../conf/config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.cuda) else "cpu")
    log.info("DEVICE : {}".format(device))

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg.enable_cudnn

    # Checkpoint
    checkpoint = ModelCheckpoint(cfg.checkpoint_dir, cfg.model_name, cfg.weight_name, strict=True)

    # Setup the dataset config
    # Generic config

    dataset = instantiate_dataset(cfg.data)
    model = checkpoint.create_model(dataset, weight_name=cfg.weight_name)
    log.info(model)
    log.info("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

    log.info(dataset)

    model.eval()
    if cfg.enable_dropout:
        model.enable_dropout_in_eval()
    model = model.to(device)

    # Run training / evaluation
    output_path = os.path.join(cfg.checkpoint_dir, cfg.data.name, "features")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    run(model, dataset, device, output_path, cfg)


if __name__ == "__main__":
    main()
