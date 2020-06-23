import os
import torch
import hydra
import logging
import numpy as np
from omegaconf import OmegaConf
import pickle

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset
from torch_points3d.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from torch_points3d.utils.colors import COLORS
from torch_points3d.utils.config import determine_stage
from torch_points3d.visualization import Visualizer
from torch_points3d.utils.config import set_debugging_vars_to_global
from torch_points3d.utils.debugging_vars import extract_histogram

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


def process(model, data, device):
    with torch.no_grad():
        model.set_input(data, device)
        model.forward()


def run_epoch(model: BaseModel, loader, device: str, num_batches: int):
    model.eval()
    with Ctq(loader) as tq_loader:
        for batch_idx, data in enumerate(tq_loader):
            if batch_idx < num_batches:
                process(model, data, device)
            else:
                break


def run(cfg, model: BaseModel, dataset: BaseDataset, device, measurement_name: str):
    measurements = {}

    num_batches = getattr(cfg.debugging, "num_batches", np.inf)

    run_epoch(model, dataset.train_dataloader, device, num_batches)
    measurements["train"] = extract_histogram(model.get_spatial_ops(), normalize=False)

    if dataset.has_val_loader:
        run_epoch(model, dataset.val_dataloader, device, num_batches)
        measurements["val"] = extract_histogram(model.get_spatial_ops(), normalize=False)

    for loader in dataset.test_dataloaders:
        run_epoch(model, dataset.test_dataloaders, device, num_batches)
        measurements[loader.dataset.name] = extract_histogram(model.get_spatial_ops(), normalize=False)

    with open(os.path.join(DIR, "measurements/{}.pickle".format(measurement_name)), "wb") as f:
        pickle.dump(measurements, f)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(cfg.pretty())

    set_debugging_vars_to_global(cfg.debugging)

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.training.cuda) else "cpu")
    log.info("DEVICE : {}".format(device))

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg.training.enable_cudnn

    dataset = instantiate_dataset(cfg.data)
    model = instantiate_model(cfg, dataset)

    log.info(model)
    log.info("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

    # Set dataloaders
    dataset.create_dataloaders(
        model,
        cfg.training.batch_size,
        cfg.training.shuffle,
        cfg.training.num_workers,
        cfg.training.precompute_multi_scale,
    )
    log.info(dataset)

    # Run training / evaluation
    model = model.to(device)

    measurement_name = "{}_{}".format(cfg.model_name, dataset.__class__.__name__)
    run(cfg, model, dataset, device, measurement_name)


if __name__ == "__main__":
    main()
