import os
import hydra
from omegaconf import OmegaConf
import urllib.request
import logging
from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
import numpy as np

log = logging.getLogger(__name__)

from torch_points3d.trainer import Trainer

CV_S3DIS_DIR = "cv_s3dis_models"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), CV_S3DIS_DIR)

POINTNET_2_URL_MODELS = {
    "1": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1e1p0csk/pointnet2_largemsg.pt",
    "2": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2i499g2e/pointnet2_largemsg.pt",
    "3": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1e1p0csk/pointnet2_largemsg.pt",
    "4": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1ejjs4s2/pointnet2_largemsg.pt",
    "5": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/etxij0j6/pointnet2_largemsg.pt",
    "6": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/8n8t391d/pointnet2_largemsg.pt",
}

MODELS_URL = {"pointnet2": POINTNET_2_URL_MODELS}


def download_file(url, out_file):
    if not os.path.exists(out_file):
        urllib.request.urlretrieve(url, out_file)
    else:
        log.warning("WARNING: skipping download of existing file " + out_file)


def log_confusion_matrix(conf):
    log.info("====================================================")
    log.info("NUM POINTS : {}".format(np.sum(conf.confusion_matrix)))
    log.info("OA: {}".format(100 * conf.get_overall_accuracy()))
    log.info("MACC: {}".format(100 * conf.get_mean_class_accuracy()))
    log.info("MIOU : {}".format(100 * conf.get_average_intersection_union()))
    log.info("====================================================")


@hydra.main(config_path="../conf/eval.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    log.info(cfg.pretty())

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    cfg.checkpoint_dir = CHECKPOINT_DIR
    cfg.tracker_options.full_res = True

    for fold, url in MODELS_URL[cfg.model_name].items():
        download_file(url, os.path.join(CHECKPOINT_DIR, "{}_{}.pt".format(cfg.model_name, fold)))

    model_lists = os.listdir(CHECKPOINT_DIR)

    conf_paths = []
    for model_name in model_lists:
        cfg.model_name = model_name.replace(".pt", "")
        cfg.tracker_options.full_res = True
        trainer = Trainer(cfg)
        trainer.eval(stage_name="test")

        conf_path = os.path.join(CHECKPOINT_DIR, "{}.npy".format(cfg.model_name))
        np.save(conf_path, trainer._tracker.full_confusion_matrix)
        conf_paths.append(conf_path)

    confusion_matrix = ConfusionMatrix.create_from_matrix(np.sum([np.load(p) for p in conf_paths], axis=0))
    log_confusion_matrix(confusion_matrix)


if __name__ == "__main__":
    # EXAMPLE: python scripts/cv_s3dis.py checkpoint_dir={PATH_TO_TORCHPOINTS3D} model_name=pointnet2
    main()
