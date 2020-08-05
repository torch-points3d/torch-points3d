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
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), CV_S3DIS_DIR)

POINTNET_2_URL_MODELS = {
    "1": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1e1p0csk/pointnet2_largemsg.pt",
    "2": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2i499g2e/pointnet2_largemsg.pt",
    "3": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1gyokj69/pointnet2_largemsg.pt",
    "4": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1ejjs4s2/pointnet2_largemsg.pt",
    "5": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/etxij0j6/pointnet2_largemsg.pt",
    "6": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/8n8t391d/pointnet2_largemsg.pt",
}

RSCONV_URL_MODELS = {
    "1": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2b99o12e/RSConv_MSN_S3DIS.pt",
    "2": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1onl4h59/RSConv_MSN_S3DIS.pt",
    "3": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2cau6jua/RSConv_MSN_S3DIS.pt",
    "4": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1qqmzgnz/RSConv_MSN_S3DIS.pt",
    "5": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/378enxsu/RSConv_MSN_S3DIS.pt",
    "6": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/23f4upgc/RSConv_MSN_S3DIS.pt",
}

KPCONV_URL_MODELS = {
    "1": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/okiba8gp/KPConvPaper.pt",
    "2": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2at56wrm/KPConvPaper.pt",
    "3": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1ipv9lso/KPConvPaper.pt",
    "4": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2c13jhi0/KPConvPaper.pt",
    "5": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1kf8yg5s/KPConvPaper.pt",
    "6": "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2ph7ejss/KPConvPaper.pt",
}

MINKOWSKI_URL_MODELS = {
    "1": "https://api.wandb.ai/files/nicolas/s3dis-benchmark/1fyr7ri9/Res16UNet34C.pt",
    "2": "https://api.wandb.ai/files/nicolas/s3dis-benchmark/1gdgx2ni/Res16UNet34C.pt",
    "3": "https://api.wandb.ai/files/nicolas/s3dis-benchmark/gt3ttamp/Res16UNet34C.pt",
    "4": "https://api.wandb.ai/files/nicolas/s3dis-benchmark/36yxu3yc/Res16UNet34C.pt",
    "5": "https://api.wandb.ai/files/nicolas/s3dis-benchmark/2r0tsub1/Res16UNet34C.pt",
    "6": "https://api.wandb.ai/files/nicolas/s3dis-benchmark/30yrkk5p/Res16UNet34C.pt",
}

MODELS_URL = {
    "pointnet2": POINTNET_2_URL_MODELS,
    "rsconv": RSCONV_URL_MODELS,
    "kpconv": KPCONV_URL_MODELS,
    "minkowski": MINKOWSKI_URL_MODELS,
}


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
    workdir = os.path.join(BASE_DIR, cfg.model_name)
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    cfg.checkpoint_dir = workdir
    cfg.tracker_options.full_res = True
    local_models = {}
    for fold, url in MODELS_URL[cfg.model_name].items():
        local_file = os.path.join(workdir, "{}_{}.pt".format(cfg.model_name, fold))
        local_models[fold] = local_file
        download_file(url, local_file)

    conf_paths = []
    for fold, model_name in local_models.items():
        cfg.model_name = model_name.replace(".pt", "")
        cfg.tracker_options.full_res = True
        trainer = Trainer(cfg)
        assert str(trainer._checkpoint.data_config.fold) == fold
        trainer.eval(stage_name="test")

        conf_path = os.path.join(workdir, "{}.npy".format(cfg.model_name))
        np.save(conf_path, trainer._tracker.full_confusion_matrix.get_confusion_matrix())
        del trainer
        conf_paths.append(conf_path)

    confusion_matrix = ConfusionMatrix.create_from_matrix(np.sum([np.load(p) for p in conf_paths], axis=0))
    log_confusion_matrix(confusion_matrix)


if __name__ == "__main__":
    # EXAMPLE: python scripts/cv_s3dis.py checkpoint_dir=`pwd` model_name=pointnet2
    main()
