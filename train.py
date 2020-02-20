import os
import torch
import hydra
import time
import logging
from omegaconf import OmegaConf

# Import building function for model and dataset
from src.datasets import instantiate_dataset
from src.models import instantiate_model

# Import BaseModel / BaseDataset for type checking
from src.models.base_model import BaseModel
from src.datasets.base_dataset import BaseDataset

# Import from metrics
from src.metrics.base_tracker import BaseTracker
from src.metrics.colored_tqdm import Coloredtqdm as Ctq
from src.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from src.utils.colors import COLORS
from src.utils.config import determine_stage, launch_wandb
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
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(cfg.pretty())

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.training.cuda) else "cpu")
    log.info("DEVICE : {}".format(device))

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg.training.enable_cudnn

    # Start Wandb if public
    launch_wandb(cfg, cfg.wandb.public and cfg.wandb.log)

    # Checkpoint
    checkpoint = ModelCheckpoint(
        cfg.training.checkpoint_dir,
        cfg.model_name,
        cfg.training.weight_name,
        run_config=cfg,
        resume=bool(cfg.training.checkpoint_dir),
    )

    # Create model and datasets
    if not checkpoint.is_empty and cfg.training.resume:
        dataset = instantiate_dataset(checkpoint.data_config)
        model = checkpoint.create_model(dataset, weight_name=cfg.training.weight_name)
    else:
        dataset = instantiate_dataset(cfg.data)
        model = instantiate_model(cfg, dataset)
        model.instantiate_optimizers(cfg)
    log.info(model)
    model.log_optimizers()
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

    # Choose selection stage
    selection_stage = determine_stage(cfg, dataset.has_val_loader)
    checkpoint.selection_stage = selection_stage
    tracker: BaseTracker = dataset.get_tracker(model, dataset, cfg.wandb.log, cfg.tensorboard.log)

    launch_wandb(cfg, not cfg.wandb.public and cfg.wandb.log)

    # Run training / evaluation
    model = model.to(device)
    visualizer = Visualizer(cfg.visualization, dataset.num_batches, dataset.batch_size, os.getcwd())
    run(cfg, model, dataset, device, tracker, checkpoint, visualizer)


if __name__ == "__main__":
    main()
