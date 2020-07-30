"""
compute features, evaluate metrics and save results
only axcept fragment
"""

import open3d
import torch
import hydra
import logging
from omegaconf import OmegaConf
import os
import os.path as osp
import sys
import pandas as pd

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..", "..")
sys.path.insert(0, ROOT)

from torch_points3d.models.model_factory import instantiate_model
from torch_points3d.datasets.dataset_factory import instantiate_dataset, get_dataset_class
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

from torch_points3d.metrics.registration_metrics import estimate_transfo
from torch_points3d.metrics.registration_metrics import compute_metrics
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

log = logging.getLogger(__name__)


def torch2o3d(xyz, color=[1, 0, 0]):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    pcd.paint_uniform_color(color)
    return pcd


def run(model: BaseModel, dataset: BaseDataset, device, cfg):
    dataset.create_dataloaders(
        model, 1, False, cfg.training.num_workers, False,
    )
    loader = dataset.test_dataloaders[0]
    list_res = []
    with Ctq(loader) as tq_test_loader:
        for i, data in enumerate(tq_test_loader):
            with torch.no_grad():
                model.set_input(data, device)
                model.forward()

                name_scene, name_pair_source, name_pair_target = dataset.test_dataset[0].get_name(i)
                input, input_target = model.get_input()
                xyz, xyz_target = input.pos, input_target.pos
                ind, ind_target = input.ind, input_target.ind
                matches_gt = torch.stack([ind, ind_target]).transpose(0, 1)
                feat, feat_target = model.get_output()
                rand = torch.randperm(len(feat))[: cfg.data.num_points]
                rand_target = torch.randperm(len(feat_target))[: cfg.data.num_points]
                res = dict(name_scene=name_scene, name_pair_source=name_pair_source, name_pair_target=name_pair_target)
                T_gt = estimate_transfo(xyz[matches_gt[:, 0]], xyz_target[matches_gt[:, 1]])
                metric = compute_metrics(
                    xyz[rand],
                    xyz_target[rand_target],
                    feat[rand],
                    feat_target[rand_target],
                    T_gt,
                    sym=cfg.data.sym,
                    tau_1=cfg.data.tau_1,
                    tau_2=cfg.data.tau_2,
                    rot_thresh=cfg.data.rot_thresh,
                    trans_thresh=cfg.data.trans_thresh,
                    use_ransac=cfg.data.use_ransac,
                    ransac_thresh=cfg.data.first_subsampling,
                    use_teaser=cfg.data.use_teaser,
                    noise_bound_teaser=cfg.data.noise_bound_teaser,
                )

                res = dict(**res, **metric)
                print(res)
                list_res.append(res)
                # pcd = torch2o3d(input.pos, color=[1,0,0])
                # pcd_t = torch2o3d(input_target.pos, color=[0,1,0])
                # open3d.visualization.draw_geometries([pcd, pcd_t])
                # open3d.visualization.draw_geometries([pcd.transform(T_gt.detach().cpu().numpy()),pcd_t])
    df = pd.DataFrame(list_res)
    output_path = os.path.join(cfg.training.checkpoint_dir, cfg.data.name, "matches")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    df.to_csv(osp.join(output_path, "final_res.csv"))
    print(df.groupby("name_scene").mean())


@hydra.main(config_path="../../conf/config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.training.cuda) else "cpu")
    log.info("DEVICE : {}".format(device))

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg.training.enable_cudnn

    # Checkpoint
    checkpoint = ModelCheckpoint(cfg.training.checkpoint_dir, cfg.model_name, cfg.training.weight_name, strict=True)

    # Setup the dataset config
    # Generic config

    dataset = instantiate_dataset(cfg.data)
    model = checkpoint.create_model(dataset, weight_name=cfg.training.weight_name)
    log.info(model)
    log.info("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

    log.info(dataset)

    model.eval()
    if cfg.enable_dropout:
        model.enable_dropout_in_eval()
    model = model.to(device)

    run(model, dataset, device, cfg)


if __name__ == "__main__":
    main()
