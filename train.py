import os
import torch
import hydra
import time
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
from src.metrics.model_checkpoint import get_model_checkpoint, ModelCheckpoint

# Utils import
from src.utils.model_building_utils.model_definition_resolver import resolve_model
from src.utils.colors import COLORS
from src.utils.config import set_format, determine_stage, launch_wandb
from src.visualization import Visualizer

log = logging.getLogger(__name__)


def train_epoch(
    epoch: int,
    model: BaseModel,
    dataset,
    device: str,
    tracker: BaseTracker,
    checkpoint: ModelCheckpoint,
    visualizer: Visualizer,
):
    model.train()
    tracker.reset("train")
    visualizer.reset(epoch, "train")
    train_loader = dataset.train_dataloader()

    iter_data_time = time.time()
    with Ctq(train_loader) as tq_train_loader:
        for i, data in enumerate(tq_train_loader):
            data = data.to(device)  # This takes time

            model.set_input(data)
            t_data = time.time() - iter_data_time

            iter_start_time = time.time()
            model.optimize_parameters(epoch, dataset.batch_size)
            if i % 10 == 0:
                tracker.track(model)

            tq_train_loader.set_postfix(
                **tracker.get_metrics(),
                data_loading=float(t_data),
                iteration=float(time.time() - iter_start_time),
                color=COLORS.TRAIN_COLOR
            )

            if visualizer.is_active:
                visualizer.save_visuals(model.get_current_visuals())

            iter_data_time = time.time()

    metrics = tracker.publish()
    checkpoint.save_best_models_under_current_metrics(model, metrics)
    log.info("Learning rate = %f" % model.learning_rate)


def eval_epoch(
    epoch: int,
    model: BaseModel,
    dataset,
    device,
    tracker: BaseTracker,
    checkpoint: ModelCheckpoint,
    visualizer: Visualizer,
):
    model.eval()
    tracker.reset("val")
    visualizer.reset(epoch, "val")
    loader = dataset.val_dataloader()
    with Ctq(loader) as tq_val_loader:
        for data in tq_val_loader:
            data = data.to(device)
            with torch.no_grad():
                model.set_input(data)
                model.forward()

            tracker.track(model)
            tq_val_loader.set_postfix(**tracker.get_metrics(), color=COLORS.VAL_COLOR)

            if visualizer.is_active:
                visualizer.save_visuals(model.get_current_visuals())

    metrics = tracker.publish()
    tracker.print_summary()
    checkpoint.save_best_models_under_current_metrics(model, metrics)


def test_epoch(
    epoch: int,
    model: BaseModel,
    dataset,
    device,
    tracker: BaseTracker,
    checkpoint: ModelCheckpoint,
    visualizer: Visualizer,
):
    model.eval()
    tracker.reset("test")
    visualizer.reset(epoch, "test")
    loader = dataset.test_dataloader()
    with Ctq(loader) as tq_test_loader:
        for data in tq_test_loader:
            data = data.to(device)
            with torch.no_grad():
                model.set_input(data)
                model.forward()

            tracker.track(model)
            tq_test_loader.set_postfix(**tracker.get_metrics(), color=COLORS.TEST_COLOR)

            if visualizer.is_active:
                visualizer.save_visuals(model.get_current_visuals())

    metrics = tracker.publish()
    tracker.print_summary()
    checkpoint.save_best_models_under_current_metrics(model, metrics)


def run(
    cfg, model, dataset: BaseDataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint, visualizer: Visualizer
):
    for epoch in range(checkpoint.start_epoch, cfg.training.epochs):
        log.info("EPOCH %i / %i", epoch, cfg.training.epochs)
        train_epoch(epoch, model, dataset, device, tracker, checkpoint, visualizer)
        if dataset.has_val_loader:
            eval_epoch(epoch, model, dataset, device, tracker, checkpoint, visualizer)

        test_epoch(epoch, model, dataset, device, tracker, checkpoint, visualizer)

    # Single test evaluation in resume case
    if checkpoint.start_epoch > cfg.training.epochs:
        test_epoch(epoch, model, dataset, device, tracker, checkpoint, visualizer)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    if cfg.pretty_print:
        print(cfg.pretty())

    # Get task and model_name
    tested_task = cfg.data.task
    tested_model_name = cfg.model_name

    # Find configs
    model_config = getattr(cfg.models, tested_model_name, None)
    dataset_config = cfg.data
    cfg_training = set_format(model_config, cfg.training)
    model_class = getattr(model_config, "class")
    tested_dataset_class = getattr(dataset_config, "class")
    otimizer_class = getattr(cfg.training.optim.optimizer, "class")
    scheduler_class = getattr(cfg.lr_scheduler, "class")
    wandb_public = getattr(cfg.wandb, "public", False)

    # Define tags for Wandb
    tags = [
        tested_model_name,
        model_class.split(".")[0],
        tested_dataset_class,
        otimizer_class,
        scheduler_class,
    ]

    launch_wandb(cfg, tags, wandb_public and cfg.wandb.log)

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.training.cuda) else "cpu")
    log.info("DEVICE : {}".format(device))

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg_training.enable_cudnn

    # Find and create associated dataset
    dataset_config.dataroot = hydra.utils.to_absolute_path(dataset_config.dataroot)
    dataset = instantiate_dataset(tested_dataset_class, tested_task, dataset_config, cfg_training)

    # Create visualizer
    visualizer = Visualizer(cfg.visualization, dataset.num_batches, dataset.batch_size, os.getcwd())

    # Find and create associated model
    resolve_model(model_config, dataset, tested_task)
    model_config = OmegaConf.merge(model_config, cfg_training)
    model = instantiate_model(model_class, tested_task, model_config, dataset)

    # Log model
    log.info(model)

    # Initialize optimizer, schedulers
    model.instantiate_optimizers(cfg)

    # Choose selection stage
    selection_stage = determine_stage(cfg, dataset.has_val_loader)

    # Set sampling / search strategies
    if cfg_training.precompute_multi_scale:
        dataset.set_strategies(model)

    log.info("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

    tracker: BaseTracker = dataset.get_tracker(model, tested_task, dataset, cfg.wandb, cfg.tensorboard)

    checkpoint = get_model_checkpoint(
        model,
        cfg_training.checkpoint_dir,
        tested_model_name,
        cfg_training.resume,
        cfg_training.weight_name,
        selection_stage,
    )

    launch_wandb(cfg, tags, not wandb_public and cfg.wandb.log)

    # Run training / evaluation
    model = model.to(device)
    run(cfg, model, dataset, device, tracker, checkpoint, visualizer)


if __name__ == "__main__":
    main()
