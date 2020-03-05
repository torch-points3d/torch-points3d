import torch
import hydra
import logging
from omegaconf import OmegaConf
import os
import sys
import numpy as np


DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

# Import building function for model and dataset
from src.datasets.dataset_factory import instantiate_dataset, get_dataset_class
from src.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from src.models.base_model import BaseModel
from src.datasets.base_dataset import BaseDataset

# Import from metrics
from src.metrics.colored_tqdm import Coloredtqdm as Ctq
from src.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from src.utils.colors import COLORS

log = logging.getLogger(__name__)


def save(prefix, predicted):
    for key, value in predicted.items():
        filename = key.split(".")[0]
        out_file = filename + "_pred"
        np.save(os.path.join(prefix, out_file), value)


def run(model: BaseModel, dataset: BaseDataset, device, output_path):
    loaders = dataset.test_dataloaders()
    predicted = {}
    for idx, loader in enumerate(loaders):
        dataset.get_test_dataset_name(idx)
        with Ctq(loader) as tq_test_loader:
            for data in tq_test_loader:
                data = data.to(device)
                with torch.no_grad():
                    model.set_input(data)
                    model.forward()
                predicted = {**predicted, **dataset.predict_original_samples(data, model.conv_type, model.get_output())}

    save(output_path, predicted)


@hydra.main(config_path="conf/config.yaml")
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
    train_dataset_cls = get_dataset_class(checkpoint.data_config)
    setattr(checkpoint.data_config, "class", train_dataset_cls.FORWARD_CLASS)
    setattr(checkpoint.data_config, "dataroot", cfg.input_path)

    # Datset specific configs
    if cfg.data:
        for key, value in cfg.data.items():
            checkpoint.data_config.update(key, value)

    # Create dataset and mdoel
    dataset = instantiate_dataset(checkpoint.data_config)
    model = checkpoint.create_model(dataset, weight_name=cfg.weight_name)
    log.info(model)
    log.info("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

    # Set dataloaders
    dataset.create_dataloaders(
        model, cfg.batch_size, cfg.shuffle, cfg.num_workers, False,
    )
    log.info(dataset)

    model.eval()
    if cfg.enable_dropout:
        model.enable_dropout_in_eval()
    model = model.to(device)

    # Run training / evaluation
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)

    run(model, dataset, device, cfg.output_path)


if __name__ == "__main__":
    main()
