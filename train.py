import torch
import numpy as np
import hydra
import time
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


def train_epoch(epoch, model: BaseModel, dataset, device: str, tracker: BaseTracker, checkpoint: ModelCheckpoint, log):
    model.train()
    tracker.reset("train")
    train_loader = dataset.train_dataloader()

    iter_data_time = time.time()
    with Ctq(train_loader) as tq_train_loader:
        for i, data in enumerate(tq_train_loader):
            data = data.to(device)  # This takes time

            model.set_input(data)
            t_data = time.time() - iter_data_time

            iter_start_time = time.time()
            model.optimize_parameters(dataset.batch_size)

            if i % 10 == 0:
                tracker.track(model)

            tq_train_loader.set_postfix(
                **tracker.get_metrics(),
                data_loading=float(t_data),
                iteration=float(time.time() - iter_start_time),
                color=COLORS.TRAIN_COLOR
            )
            iter_data_time = time.time()

    metrics = tracker.publish()
    checkpoint.save_best_models_under_current_metrics(model, metrics)
    log.info("Learning rate = %f" % model.learning_rate)


def eval_epoch(model: BaseModel, dataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint, log):
    model.eval()
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

    metrics = tracker.publish()
    tracker.print_summary()
    checkpoint.save_best_models_under_current_metrics(model, metrics)


def test_epoch(model: BaseModel, dataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint, log):
    model.eval()
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

    metrics = tracker.publish()
    tracker.print_summary()
    checkpoint.save_best_models_under_current_metrics(model, metrics)


def run(cfg, model, dataset: BaseDataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint, log):
    for epoch in range(checkpoint.start_epoch, cfg.training.epochs):
        log.info("EPOCH %i / %i", epoch, cfg.training.epochs)
        train_epoch(epoch, model, dataset, device, tracker, checkpoint, log)

        if dataset.has_val_loader:
            eval_epoch(model, dataset, device, tracker, checkpoint, log)

        test_epoch(model, dataset, device, tracker, checkpoint, log)

    # Single test evaluation in resume case
    if checkpoint.start_epoch > cfg.training.epochs:
        test_epoch(model, dataset, device, tracker, checkpoint, log)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    log = logging.getLogger(__name__)
    print(cfg.pretty())

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.training.cuda) else "cpu")
    print("DEVICE : {}".format(device))

    # Get task and model_name
    tested_task = cfg.data.task
    tested_model_name = cfg.model_name

    # Find and create associated model
    model_config = getattr(cfg.models, tested_model_name, None)

    # Find which dataloader to use
    cfg_training = set_format(model_config, cfg.training)

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg_training.enable_cudnn

    # Find and create associated dataset
    dataset_config = cfg.data
    tested_dataset_name = dataset_config.name
    dataset_config.dataroot = hydra.utils.to_absolute_path(dataset_config.dataroot)
    dataset = find_dataset_using_name(tested_dataset_name, tested_task)(dataset_config, cfg_training)

    # Find and create associated model
    resolve_model(model_config, dataset, tested_task)
    model_config = OmegaConf.merge(model_config, cfg_training, dataset_config)
    model = find_model_using_name(model_config.architecture, tested_task, model_config, dataset)

    log.info(model)

    # Optimizer
    lr_params = cfg_training.learning_rate
    model.set_optimizer(getattr(torch.optim, cfg_training.optimizer, None), lr_params=lr_params)

    # Set sampling / search strategies
    dataset.set_strategies(model, precompute_multi_scale=cfg_training.precompute_multi_scale)

    model = model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    log.info("Model size = %i", params)

    # metric tracker
    if cfg.wandb.log:
        import wandb

        wandb.init(project=cfg.wandb.project)
        # wandb.watch(model)

    tracker: BaseTracker = dataset.get_tracker(model, tested_task, dataset, cfg.wandb, cfg.tensorboard)

    checkpoint = get_model_checkpoint(
        model,
        cfg_training.checkpoint_dir,
        tested_model_name,
        cfg_training.resume,
        cfg_training.weight_name,
        "val" if dataset.has_val_loader else "test",
    )

    # Run training / evaluation
    run(cfg, model, dataset, device, tracker, checkpoint, log)


if __name__ == "__main__":
    main()
