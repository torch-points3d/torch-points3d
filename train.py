from os import path as osp

import torch
torch.backends.cudnn.enabled = False
import numpy as np
import torch.nn.functional as F
from datasets.utils import find_dataset_using_name
import hydra
from torch_geometric.utils import intersection_and_union as i_and_u
from models.utils import find_model_using_name
from tqdm import tqdm as tq
import time
import wandb

from visualizer import print_current_losses
#wandb.init(project="dpc-benchmark")


def train(epoch, model, train_loader, device, options):
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    iter_data_time = time.time()
    for i, data in tq(enumerate(train_loader)):
        iter_start_time = time.time()  # timer for computation per iteration
        t_data = iter_start_time - iter_data_time

        data = data.to(device)
        model.set_input(data)
        model.optimize_parameters()
        if i % options.training.print_frequency == 0:
            print_current_losses(epoch, i, model.get_current_losses(), time.time() - iter_start_time, t_data)
        correct_nodes += model.get_output().max(dim=1)[1].eq(data.y).sum().item()
        total_nodes += data.num_nodes
        iter_data_time = time.time()

        #uncomment to print loss and accurancy every 10 batches - to check if model is training correctly 
        if (i + 1) % 10 == 0:
            print('[{}/{}] Loss: {:.4f}, Train Accuracy: {:.4f}'.format(
            i + 1, len(train_loader), total_loss / 10,
            correct_nodes / total_nodes))
            total_loss = correct_nodes = total_nodes = 0

    
    wandb.log({"Train Accuracy": correct_nodes / total_nodes})


def test(model, loader, num_classes, device):
    model.eval()

    correct_nodes = total_nodes = 0
    intersections, unions, categories = [], [], []
    for data in tq(loader):
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        pred = out.max(dim=1)[1]
        correct_nodes += pred.eq(data.y).sum().item()
        total_nodes += data.num_nodes
        i, u = i_and_u(pred, data.y, num_classes, data.batch)
        intersections.append(i.to(device))
        unions.append(u.to(device))
        categories.append(data.category.to(device))

    category = torch.cat(categories, dim=0)
    intersection = torch.cat(intersections, dim=0)
    union = torch.cat(unions, dim=0)

    ious = [[] for _ in range(len(loader.dataset.categories))]
    for j in range(len(loader.dataset)):
        i = intersection[j, loader.dataset.y_mask[category[j]]]
        u = union[j, loader.dataset.y_mask[category[j]]]
        iou = i.to(torch.float) / u.to(torch.float)
        iou[torch.isnan(iou)] = 1
        ious[category[j]].append(iou.mean().item())

    for cat in range(len(loader.dataset.categories)):
        ious[cat] = torch.tensor(ious[cat]).mean().item()

    return correct_nodes / total_nodes, torch.tensor(ious).mean().item()


def run(cfg, model, dataset, device):
    train_loader = dataset.train_dataloader()
    test_loader = dataset.test_dataloader()
    for epoch in range(1, 31):
        train(epoch, model, train_loader, device, cfg)
        acc, iou = test(model, test_loader, dataset.num_classes, device)
        wandb.log({"Test Accuracy": acc, "Test IoU": iou})
        print('Epoch: {:02d}, Acc: {:.4f}, IoU: {:.4f}'.format(epoch, acc, iou))


@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    
    # GET ARGUMENTS
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
    model = find_model_using_name(model_config.type, tested_task, model_config, dataset.num_classes)
    model.set_optimizer(torch.optim.Adam)

    # wandb.watch(model)
    model = model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model size = %i" % params)

    # Run training / evaluation
    run(cfg, model, dataset, device)


if __name__ == "__main__":
    main()
