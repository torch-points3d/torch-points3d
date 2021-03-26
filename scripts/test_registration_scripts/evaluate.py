"""
compute features, evaluate metrics and save results
only axcept fragment
"""
import copy
import open3d
import torch
import hydra
import logging
from omegaconf import OmegaConf
import os
import os.path as osp
import sys
import pandas as pd
import time


DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..", "..")
sys.path.insert(0, ROOT)

from torch_points3d.models.model_factory import instantiate_model
from torch_points3d.datasets.dataset_factory import instantiate_dataset, get_dataset_class
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

from torch_points3d.utils.registration import (
    estimate_transfo,
    teaser_pp_registration,
    fast_global_registration,
    get_matches,
)
from torch_points3d.metrics.registration_metrics import (
    compute_hit_ratio,
    compute_transfo_error,
    compute_scaled_registration_error,
    compute_registration_recall,
)

from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

log = logging.getLogger(__name__)


def compute_metrics(
    xyz,
    xyz_target,
    feat,
    feat_target,
    T_gt,
    sym=False,
    tau_1=0.1,
    tau_2=0.05,
    rot_thresh=5,
    trans_thresh=5,
    use_ransac=False,
    ransac_thresh=0.02,
    use_teaser=False,
    noise_bound_teaser=0.1,
    registration_recall_thresh=0.2,
    xyz_gt=None,
    xyz_target_gt=None,
):
    """
    compute all the necessary metrics
    compute the hit ratio,
    compute the feat_match_ratio

    using fast global registration
    compute the translation error
    compute the rotation error
    compute rre, and compute the rte

    using ransac
    compute the translation error
    compute the rotation error
    compute rre, and compute the rte

    using Teaser++
    compute the translation error
    compute the rotation error
    compute rre, and compute the rtr

    Parameters
    ----------

    xyz: torch tensor of size N x 3
    xyz_target: torch tensor of size N x 3

    feat: torch tensor of size N x C
    feat_target: torch tensor of size N x C


    T_gt; 4 x 4 matrix
    """

    res = dict()
    t0 = time.time()
    matches_pred = get_matches(feat, feat_target, sym=sym)
    t1 = time.time()
    hit_ratio = compute_hit_ratio(xyz[matches_pred[:, 0]], xyz_target[matches_pred[:, 1]], T_gt, tau_1)
    res["hit_ratio"] = hit_ratio.item()
    res["feat_match_ratio"] = float(hit_ratio.item() > tau_2)
    res["time_get_matches"] = t1 - t0
    # fast global registration
    t = time.time()
    T_fgr = fast_global_registration(xyz[matches_pred[:, 0]], xyz_target[matches_pred[:, 1]])
    trans_error_fgr, rot_error_fgr = compute_transfo_error(T_fgr, T_gt)
    res["trans_error_fgr"] = trans_error_fgr.item()
    res["rot_error_fgr"] = rot_error_fgr.item()
    res["rre_fgr"] = float(rot_error_fgr.item() < rot_thresh)
    res["rte_fgr"] = float(trans_error_fgr.item() < trans_thresh)
    res["sr_fgr"] = compute_scaled_registration_error(xyz, T_gt, T_fgr).item()
    res["time_fgr"] = time.time() - t
    if xyz_gt is not None and xyz_target_gt is not None:
        res["registration_recall_fgr"] = compute_registration_recall(
            xyz_gt, xyz_target_gt, T_fgr, registration_recall_thresh
        )

    # teaser pp
    if use_teaser:
        t = time.time()
        T_teaser = teaser_pp_registration(
            xyz[matches_pred[:, 0]], xyz_target[matches_pred[:, 1]], noise_bound=noise_bound_teaser
        )
        trans_error_teaser, rot_error_teaser = compute_transfo_error(T_teaser, T_gt)
        res["trans_error_teaser"] = trans_error_teaser.item()
        res["rot_error_teaser"] = rot_error_teaser.item()
        res["rre_teaser"] = float(rot_error_teaser.item() < rot_thresh)
        res["rte_teaser"] = float(trans_error_teaser.item() < trans_thresh)
        res["sr_teaser"] = compute_scaled_registration_error(xyz, T_gt, T_teaser).item()
        res["time_teaser"] = time.time() - t
        if xyz_gt is not None and xyz_target_gt is not None:
            res["registration_recall_teaser"] = compute_registration_recall(
                xyz_gt, xyz_target_gt, T_teaser, registration_recall_thresh
            )

    if use_ransac:
        raise NotImplementedError

    return res


def run(model: BaseModel, dataset: BaseDataset, device, cfg):

    reg_thresh = cfg.data.registration_recall_thresh
    if reg_thresh is None:
        reg_thresh = 0.2
    print(time.strftime("%Y%m%d-%H%M%S"))
    dataset.create_dataloaders(
        model, 1, False, cfg.training.num_workers, False,
    )
    loader = dataset.test_dataloaders[0]
    list_res = []
    with Ctq(loader) as tq_test_loader:
        for i, data in enumerate(tq_test_loader):
            with torch.no_grad():
                t0 = time.time()
                model.set_input(data, device)
                model.forward()
                t1 = time.time()
                name_scene, name_pair_source, name_pair_target = dataset.test_dataset[0].get_name(i)
                input, input_target = model.get_input()
                xyz, xyz_target = input.pos, input_target.pos
                ind, ind_target = input.ind, input_target.ind
                matches_gt = torch.stack([ind, ind_target]).transpose(0, 1)
                feat, feat_target = model.get_output()
                # rand = voxel_selection(xyz, grid_size=0.06, min_points=cfg.data.min_points)
                # rand_target = voxel_selection(xyz_target, grid_size=0.06, min_points=cfg.data.min_points)

                rand = torch.randperm(len(feat))[: cfg.data.num_points]
                rand_target = torch.randperm(len(feat_target))[: cfg.data.num_points]
                res = dict(name_scene=name_scene, name_pair_source=name_pair_source, name_pair_target=name_pair_target)
                T_gt = estimate_transfo(xyz[matches_gt[:, 0]], xyz_target[matches_gt[:, 1]])
                t2 = time.time()
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
                    xyz_gt=xyz[matches_gt[:, 0]],
                    xyz_target_gt=xyz_target[matches_gt[:, 1]],
                    registration_recall_thresh=reg_thresh,
                )
                res = dict(**res, **metric)
                res["time_feature"] = t1 - t0
                res["time_feature_per_point"] = (t1 - t0) / (len(input.pos) + len(input_target.pos))
                res["time_prep"] = t2 - t1

                list_res.append(res)

    df = pd.DataFrame(list_res)
    output_path = os.path.join(cfg.training.checkpoint_dir, cfg.data.name, "matches")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    df.to_csv(osp.join(output_path, "final_res_{}.csv".format(time.strftime("%Y%m%d-%H%M%S"))))
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
    if not checkpoint.is_empty:
        model = checkpoint.create_model(dataset, weight_name=cfg.training.weight_name)
    else:
        log.info("No Checkpoint for this model")
        model = instantiate_model(copy.deepcopy(cfg), dataset)
        model.set_pretrained_weights()
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
