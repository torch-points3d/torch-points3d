from os import path as osp

import torch
torch.backends.cudnn.enabled = False
import numpy as np
import torch.nn.functional as F
from datasets.utils import find_dataset_using_name
import hydra
from torch_geometric.utils import intersection_and_union as i_and_u
from tqdm import tqdm as tq
import time
import wandb
from omegaconf import OmegaConf

from models.utils import find_model_using_name
from models.base_model import BaseModel
from metrics.metrics_tracker import get_tracker, BaseTracker
from metrics.colored_tqdm import Coloredtqdm as Ctq, COLORS


def train(epoch, model: BaseModel, train_loader, device, tracker: BaseTracker):
    model.train()
    tracker.reset("train")

    iter_data_time = time.time()
    with Ctq(train_loader) as tq_train_loader:
        for data in tq_train_loader:
            iter_start_time = time.time()  # timer for computation per iteration
            t_data = iter_start_time - iter_data_time

            data = data.to(device)
            model.set_input(data)
            model.optimize_parameters()

            tracker.track(model.get_current_losses(), model.get_output(), data.y)
            iter_data_time = time.time()

            tq_train_loader.set_postfix(**tracker.get_metrics(), data_loading=float(t_data),
                                        iteration=float(time.time() - iter_start_time), color=COLORS.TRAIN_COLOR)
    tracker.publish()


def test(model: BaseModel, loader, device, tracker: BaseTracker):
    model.eval()
    tracker.reset("test")

    with Ctq(loader) as tq_test_loader:
        for data in tq_test_loader:
            data = data.to(device)
            with torch.no_grad():
                model.set_input(data)
                model.forward()

            tracker.track(model.get_current_losses(), model.get_output(), data.y)
            tq_test_loader.set_postfix(**tracker.get_metrics(), color=COLORS.TEST_COLOR)

    tracker.publish()


def run(cfg, start_epoch, model, dataset, device, tracker: BaseTracker):
    train_loader = dataset.train_dataloader()
    test_loader = dataset.test_dataloader()
    for epoch in range(start_epoch, cfg.training.epochs):
        print("EPOCH {} / {}".format(epoch, cfg.training.epochs))
        train(epoch, model, train_loader, device, tracker)
        test(model, test_loader, device, tracker)
        print()


@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    # Get device
    device = torch.device('cuda' if (torch.cuda.is_available() and cfg.training.cuda)
                          else 'cpu')

    # Get task and model_name
    tested_task = cfg.experiment.task
    tested_model_name = cfg.experiment.name
    tested_dataset_name = cfg.experiment.dataset

    # Find and create associated dataset
    dataset_config = getattr(cfg.data, tested_dataset_name, None)
    dataset_config.dataroot = hydra.utils.to_absolute_path(dataset_config.dataroot)
    dataset = find_dataset_using_name(tested_dataset_name)(dataset_config, cfg.training)

    # Find and create associated model
    model_config = getattr(cfg.models, tested_model_name, None)
    model_config = OmegaConf.merge(model_config, cfg.training)
    model = find_model_using_name(model_config.type, tested_task, model_config, dataset)
    model.set_optimizer(getattr(torch.optim, cfg.training.optimizer, None), lr=cfg.training.lr)

    # Set sampling / search strategies:
    dataset.set_strategies(model, precompute_multi_scale=cfg.training.precompute_multi_scale)

    model = model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model size = %i" % params)

    # metric tracker
    if cfg.wandb.log:
        wandb.init(project=cfg.wandb.project)
        # wandb.watch(model)
    tracker: BaseTracker = get_tracker(model, tested_task, dataset, cfg.wandb,
                                       cfg.tensorboard, cfg.experiment, cfg.training)
    tracker.track_elements(model=model)  # More elements can be tracker there by providing keys

    # Run training / evaluation
    run(cfg, tracker.start_epoch, model, dataset, device, tracker)


if __name__ == "__main__":
    main()
