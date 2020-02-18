import torch
import numpy as np
import hydra
import logging
from omegaconf import OmegaConf

# Import building function for model and dataset
from src import instantiate_model, instantiate_dataset

# Import BaseModel / BaseDataset for type checking
from src.models.base_model import BaseModel
from src.datasets.base_dataset import BaseDataset

# Import from metrics
from src.metrics.base_tracker import BaseTracker
from src.metrics.colored_tqdm import Coloredtqdm as Ctq
from src.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from src.utils.model_building_utils.model_definition_resolver import resolve_model
from src.utils.colors import COLORS
from src.utils.config import set_format

log = logging.getLogger(__name__)


def eval_epoch(model: BaseModel, dataset, device, tracker: BaseTracker):
    tracker.reset("val")
    loader = dataset.val_dataloader()
    with Ctq(loader) as tq_val_loader:
        for data in tq_val_loader:
            data = data.to(device)
            with torch.no_grad():
                model.set_input(data)
                model.forward()

            tracker.track(model)
            tq_val_loader.set_postfix(**tracker.get_metrics(), color=COLORS.VAL_COLOR)

    tracker.print_summary()


def test_epoch(model: BaseModel, dataset, device, tracker: BaseTracker):
    tracker.reset("test")
    loader = dataset.test_dataloader()
    with Ctq(loader) as tq_test_loader:
        for data in tq_test_loader:
            data = data.to(device)
            with torch.no_grad():
                model.set_input(data)
                model.forward()

            tracker.track(model)
            tq_test_loader.set_postfix(**tracker.get_metrics(), color=COLORS.TEST_COLOR)

    tracker.print_summary()


def run(cfg, model, dataset: BaseDataset, device, tracker: BaseTracker):
    if dataset.has_val_loader:
        eval_epoch(model, dataset, device, tracker)

    test_epoch(model, dataset, device, tracker)


@hydra.main(config_path="conf/eval.yaml")
def main(cfg):
    # Load model config
    checkpoint = ModelCheckpoint(cfg.eval.checkpoint_dir, False, "test")

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.eval.cuda) else "cpu")
    log.info("DEVICE : {}".format(device))

    # Get task and model_name
    tested_task = checkpoint.task

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg.eval.enable_cudnn

    # Find and create associated dataset
    dataset_config = checkpoint.data_config
    dataset_class = getattr(dataset_config, "class")
    dataset_config.dataroot = hydra.utils.to_absolute_path(dataset_config.dataroot)
    cfg_training = set_format(checkpoint.model_config, checkpoint.training_config)
    dataset = instantiate_dataset(dataset_class, tested_task, dataset_config, cfg_training)

    # Find and create associated model
    model = checkpoint.create_model_from_checkpoint(dataset, weight_name=cfg.eval.weight_name)

    log.info(model)

    model.eval()
    if cfg.eval.enable_dropout:
        model.enable_dropout_in_eval()

    # Set sampling / search strategies
    dataset.set_strategies(model)

    model = model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    log.info("Model size = %i", params)

    tracker: BaseTracker = dataset.get_tracker(model, tested_task, dataset, False, False)

    # Run training / evaluation
    run(cfg, model, dataset, device, tracker)


if __name__ == "__main__":
    main()
