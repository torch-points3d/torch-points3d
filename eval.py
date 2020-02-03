import torch
import numpy as np
import hydra
import logging
from omegaconf import OmegaConf

# Import building function for model and dataset
from src import find_model_using_name, find_dataset_using_name

# Import BaseModel / BaseDataset for type checking
from src.models.base_model import BaseModel
from src.datasets.base_dataset import BaseDataset

# Import from metrics
from src.metrics.base_tracker import BaseTracker
from src.metrics.colored_tqdm import Coloredtqdm as Ctq
from src.metrics.model_checkpoint import get_model_checkpoint, ModelCheckpoint

# Utils import
from src.utils.model_building_utils.model_definition_resolver import resolve_model
from src.utils.colors import COLORS
from src.utils.config import set_format


def eval_epoch(model: BaseModel, dataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint, log):
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


def test_epoch(model: BaseModel, dataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint, log):
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


def run(cfg, model, dataset: BaseDataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint, log):
    if dataset.has_val_loader:
        eval_epoch(model, dataset, device, tracker, checkpoint, log)

    test_epoch(model, dataset, device, tracker, checkpoint, log)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    log = logging.getLogger(__name__)

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.eval.cuda) else "cpu")
    print("DEVICE : {}".format(device))

    # Get task and model_name
    tested_task = cfg.data.task
    tested_model_name = cfg.model_name

    # Find and create associated model
    model_config = getattr(cfg.models, tested_model_name, None)

    cfg_eval = set_format(model_config, cfg.eval)

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg_eval.enable_cudnn

    # Find and create associated dataset
    dataset_config = cfg.data
    tested_dataset_name = dataset_config.name
    dataset_config.dataroot = hydra.utils.to_absolute_path(dataset_config.dataroot)
    dataset = find_dataset_using_name(tested_dataset_name, tested_task)(dataset_config, cfg_eval)

    # Find and create associated model
    resolve_model(model_config, dataset, tested_task)
    model_config = OmegaConf.merge(model_config, dataset_config)
    model = find_model_using_name(model_config.architecture, tested_task, model_config, dataset)

    log.info(model)

    model.eval()
    if cfg_eval.enable_dropout:
        model.enable_dropout_in_eval()

    # Set sampling / search strategies
    dataset.set_strategies(model, precompute_multi_scale=cfg_eval.precompute_multi_scale)

    model = model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    log.info("Model size = %i", params)

    tracker: BaseTracker = dataset.get_tracker(model, tested_task, dataset, cfg.wandb, cfg.tensorboard)

    checkpoint = get_model_checkpoint(
        model, cfg_eval.checkpoint_dir, tested_model_name, True, cfg_eval.weight_name, "test"
    )

    # Run training / evaluation
    run(cfg, model, dataset, device, tracker, checkpoint, log)


if __name__ == "__main__":
    main()
