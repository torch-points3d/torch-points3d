from os import path as osp
import torch
from torch import nn
from torch import autograd
import numpy as np
import torch.nn.functional as F
import hydra
from tqdm import tqdm as tq
import time
import wandb
from omegaconf import OmegaConf
import logging

from models.utils import find_model_using_name
from models.model_building_utils.model_definition_resolver import resolve_model
from models.base_model import BaseModel
from datasets.base_dataset import BaseDataset
from metrics.base_tracker import get_tracker, BaseTracker
from metrics.colored_tqdm import Coloredtqdm as Ctq, COLORS
from utils_folder.utils import merges_in_sub, model_fn_decorator, set_format
from metrics.model_checkpoint import get_model_checkpoint, ModelCheckpoint
from datasets.utils import find_dataset_using_name


def train(epoch, model: BaseModel, dataset, device: str, tracker: BaseTracker, checkpoint: ModelCheckpoint, log):
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
    print("Learning rate = %f" % model.learning_rate)


def test(model: BaseModel, dataset, device, tracker: BaseTracker, checkpoint: ModelCheckpoint, log):
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
        train(epoch, model, dataset, device, tracker, checkpoint, log)
        test(model, dataset, device, tracker, checkpoint, log)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    log = logging.getLogger(__name__)

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.training.cuda) else "cpu")

    # Get task and model_name
    exp = cfg.experiment
    tested_task = exp.task
    tested_model_name = exp.model_name
    tested_dataset_name = exp.dataset

    # Find and create associated model
    model_config = getattr(cfg.models, tested_model_name, None)

    # Find which dataloader to use
    cfg_training = set_format(model_config, cfg.training)

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg_training.enable_cudnn

    # Find and create associated dataset
    dataset_config = getattr(cfg.data, tested_dataset_name, None)
    dataset_config.dataroot = hydra.utils.to_absolute_path(dataset_config.dataroot)
    dataset = find_dataset_using_name(tested_dataset_name)(dataset_config, cfg_training)

    # Find and create associated model
    resolve_model(model_config, dataset, tested_task)
    model_config = merges_in_sub(model_config, [cfg_training, dataset_config])
    model = find_model_using_name(model_config.type, tested_task, model_config, dataset)

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

    tracker: BaseTracker = get_tracker(model, tested_task, dataset, cfg.wandb, cfg.tensorboard, "")

    checkpoint = get_model_checkpoint(
        model, exp.checkpoint_dir, tested_model_name, exp.resume, cfg_training.weight_name
    )

    # Run training / evaluation
    run(cfg, model, dataset, device, tracker, checkpoint, log)


if __name__ == "__main__":
    main()
