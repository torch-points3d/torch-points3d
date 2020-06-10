import os
import torch
import hydra
import time
import logging
from omegaconf import OmegaConf

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
from torch_points3d.utils.config import launch_wandb
from torch_points3d.visualization import Visualizer

log = logging.getLogger(__name__)


def train_epoch(
    epoch: int,
    model: BaseModel,
    dataset,
    device: str,
    tracker: BaseTracker,
    checkpoint: ModelCheckpoint,
    visualizer: Visualizer,
    cfg,
):

    early_break = getattr(cfg.debugging, "early_break", False)
    profiling = getattr(cfg.debugging, "profiling", False)

    model.train()
    tracker.reset("train")
    visualizer.reset(epoch, "train")
    train_loader = dataset.train_dataloader

    iter_data_time = time.time()
    with Ctq(train_loader) as tq_train_loader:
        for i, data in enumerate(tq_train_loader):
            t_data = time.time() - iter_data_time
            iter_start_time = time.time()
            model.set_input(data, device)
            model.optimize_parameters(epoch, dataset.batch_size)
            if i % 10 == 0:
                tracker.track(model, data=data, **cfg.tracker_options)

            tq_train_loader.set_postfix(
                **tracker.get_metrics(),
                data_loading=float(t_data),
                iteration=float(time.time() - iter_start_time),
                color=COLORS.TRAIN_COLOR
            )

            if visualizer.is_active:
                visualizer.save_visuals(model.get_current_visuals())

            iter_data_time = time.time()

            if early_break:
                break

            if profiling:
                if i > getattr(cfg.debugging, "num_batches", 50):
                    return 0

    tracker.finalise(**cfg.tracker_options)
    metrics = tracker.publish(epoch)
    checkpoint.save_best_models_under_current_metrics(model, metrics, tracker.metric_func)
    log.info("Learning rate = %f" % model.learning_rate)


def eval_epoch(
    epoch: int,
    model: BaseModel,
    dataset,
    device,
    tracker: BaseTracker,
    checkpoint: ModelCheckpoint,
    visualizer: Visualizer,
    cfg,
):

    early_break = getattr(cfg.debugging, "early_break", False)

    model.eval()
    tracker.reset("val")
    visualizer.reset(epoch, "val")
    loader = dataset.val_dataloader
    with Ctq(loader) as tq_val_loader:
        for data in tq_val_loader:
            with torch.no_grad():
                model.set_input(data, device)
                model.forward()

            tracker.track(model, data=data, **cfg.tracker_options)
            tq_val_loader.set_postfix(**tracker.get_metrics(), color=COLORS.VAL_COLOR)

            if visualizer.is_active:
                visualizer.save_visuals(model.get_current_visuals())

            if early_break:
                break

    tracker.finalise(**cfg.tracker_options)
    metrics = tracker.publish(epoch)
    tracker.print_summary()
    checkpoint.save_best_models_under_current_metrics(model, metrics, tracker.metric_func)


def test_epoch(
    epoch: int,
    model: BaseModel,
    dataset,
    device,
    tracker: BaseTracker,
    checkpoint: ModelCheckpoint,
    visualizer: Visualizer,
    cfg,
):
    early_break = getattr(cfg.debugging, "early_break", False)
    model.eval()

    loaders = dataset.test_dataloaders

    for loader in loaders:
        if not loader.has_labels:
            continue
        stage_name = loader.dataset.name
        tracker.reset(stage_name)
        visualizer.reset(epoch, stage_name)
        with Ctq(loader) as tq_test_loader:
            for data in tq_test_loader:
                with torch.no_grad():
                    model.set_input(data, device)
                    model.forward()

                tracker.track(model, data=data, **cfg.tracker_options)
                tq_test_loader.set_postfix(**tracker.get_metrics(), color=COLORS.TEST_COLOR)

                if visualizer.is_active:
                    visualizer.save_visuals(model.get_current_visuals())

                if early_break:
                    break

        tracker.finalise(**cfg.tracker_options)
        metrics = tracker.publish(epoch)
        tracker.print_summary()
        checkpoint.save_best_models_under_current_metrics(model, metrics, tracker.metric_func)


def run(
    cfg, model, dataset: BaseDataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint, visualizer: Visualizer
):

    profiling = getattr(cfg.debugging, "profiling", False)

    for epoch in range(checkpoint.start_epoch, cfg.training.epochs):
        log.info("EPOCH %i / %i", epoch, cfg.training.epochs)
        train_epoch(epoch, model, dataset, device, tracker, checkpoint, visualizer, cfg)
        if profiling:
            return 0
        if dataset.has_val_loader:
            eval_epoch(epoch, model, dataset, device, tracker, checkpoint, visualizer, cfg)

        if dataset.has_test_loaders:
            test_epoch(epoch, model, dataset, device, tracker, checkpoint, visualizer, cfg)

    # Single test evaluation in resume case
    if checkpoint.start_epoch > cfg.training.epochs:
        if dataset.has_test_loaders:
            test_epoch(epoch, model, dataset, device, tracker, checkpoint, visualizer, cfg)


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

    # Profiling
    profiling = getattr(cfg.debugging, "profiling", False)
    if profiling:
        # Set the num_workers as torch.utils.bottleneck doesn't work well with it
        cfg.training.num_workers = 0

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
    if not checkpoint.is_empty:
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

    # Verify attributes in dataset
    model.verify_data(dataset.train_dataset[0])

    # Choose selection stage
    selection_stage = getattr(cfg, "selection_stage", "")
    checkpoint.selection_stage = dataset.resolve_saving_stage(selection_stage)
    tracker: BaseTracker = dataset.get_tracker(cfg.wandb.log, cfg.tensorboard.log)

    launch_wandb(cfg, not cfg.wandb.public and cfg.wandb.log)

    # Run training / evaluation
    model = model.to(device)
    visualizer = Visualizer(cfg.visualization, dataset.num_batches, dataset.batch_size, os.getcwd())
    run(cfg, model, dataset, device, tracker, checkpoint, visualizer)

    # https://github.com/facebookresearch/hydra/issues/440
    hydra._internal.hydra.GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
