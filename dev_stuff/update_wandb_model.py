import argparse
import os
import os.path as osp
import torch
from omegaconf import OmegaConf
import omegaconf
import urllib.request
import wandb
import shutil
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def models():
    return [
        ("pointnet2_largemsg-s3dis-1",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1e1p0csk/pointnet2_largemsg.pt"),
        ("pointnet2_largemsg-s3dis-2",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2i499g2e/pointnet2_largemsg.pt"),
        ("pointnet2_largemsg-s3dis-3",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1gyokj69/pointnet2_largemsg.pt"),
        ("pointnet2_largemsg-s3dis-4",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1ejjs4s2/pointnet2_largemsg.pt"),
        ("pointnet2_largemsg-s3dis-5",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/etxij0j6/pointnet2_largemsg.pt"),
        ("pointnet2_largemsg-s3dis-6",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/8n8t391d/pointnet2_largemsg.pt"),
        ("pointgroup-scannet", "https://api.wandb.ai/files/nicolas/panoptic/2ta6vfu2/PointGroup.pt"),
        ("minkowski-res16-s3dis-1", "https://api.wandb.ai/files/nicolas/s3dis-benchmark/1fyr7ri9/Res16UNet34C.pt"),
        ("minkowski-res16-s3dis-2", "https://api.wandb.ai/files/nicolas/s3dis-benchmark/1gdgx2ni/Res16UNet34C.pt"),
        ("minkowski-res16-s3dis-3", "https://api.wandb.ai/files/nicolas/s3dis-benchmark/gt3ttamp/Res16UNet34C.pt"),
        ("minkowski-res16-s3dis-4", "https://api.wandb.ai/files/nicolas/s3dis-benchmark/36yxu3yc/Res16UNet34C.pt"),
        ("minkowski-res16-s3dis-5", "https://api.wandb.ai/files/nicolas/s3dis-benchmark/2r0tsub1/Res16UNet34C.pt"),
        ("minkowski-res16-s3dis-6", "https://api.wandb.ai/files/nicolas/s3dis-benchmark/30yrkk5p/Res16UNet34C.pt"),
        ("minkowski-registration-3dmatch",
         "https://api.wandb.ai/files/humanpose1/registration/2wvwf92e/MinkUNet_Fragment.pt"),
        ("minkowski-registration-kitti", "https://api.wandb.ai/files/humanpose1/KITTI/2xpy7u1i/MinkUNet_Fragment.pt"),
        ("minkowski-registration-modelnet",
         "https://api.wandb.ai/files/humanpose1/modelnet/39u5v3bm/MinkUNet_Fragment.pt"),
        ("rsconv-s3dis-1",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2b99o12e/RSConv_MSN_S3DIS.pt"),
        ("rsconv-s3dis-2",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1onl4h59/RSConv_MSN_S3DIS.pt"),
        ("rsconv-s3dis-3",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2cau6jua/RSConv_MSN_S3DIS.pt"),
        ("rsconv-s3dis-4",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1qqmzgnz/RSConv_MSN_S3DIS.pt"),
        ("rsconv-s3dis-5",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/378enxsu/RSConv_MSN_S3DIS.pt"),
        ("rsconv-s3dis-6",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/23f4upgc/RSConv_MSN_S3DIS.pt"),
        ("kpconv-s3dis-1",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/okiba8gp/KPConvPaper.pt"),
        ("kpconv-s3dis-2",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2at56wrm/KPConvPaper.pt"),
        ("kpconv-s3dis-3",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1ipv9lso/KPConvPaper.pt"),
        ("kpconv-s3dis-4",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2c13jhi0/KPConvPaper.pt"),
        ("kpconv-s3dis-5",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1kf8yg5s/KPConvPaper.pt"),
        ("kpconv-s3dis-6",
         "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/2ph7ejss/KPConvPaper.pt"),
    ]


def download_file(url, out_file):
    # If the file exists, remove it
    if os.path.exists(out_file):
        os.remove(out_file)

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    urllib.request.urlretrieve(url, out_file)


def parse_args():
    parser = argparse.ArgumentParser(
        "a simple script to convert omegaconf file to dict, you need omegaconf v 1.4.1 in order to convert files")
    parser.add_argument('-f', help='input of the .pt file', dest="file", type=str)
    parser.add_argument('--old', help='input of the .pt file', dest="old", type=str)
    parser.add_argument('-o', help='output of the .pt file', dest="out", type=str)
    args = parser.parse_args()
    return args


def convert(dico, exclude_keys=["models", "optimizer"], depth=0, verbose=True):
    if isinstance(dico, dict):
        for k, v in dico.items():
            if k not in exclude_keys:
                # print(depth * " ", k, type(v))
                convert(v, depth=depth + 1)
                if isinstance(v, omegaconf.dictconfig.DictConfig):
                    dico[k] = OmegaConf.to_container(v)

    elif isinstance(dico, list):
        for i, v in enumerate(dico):
            # print(depth * " ", i, type(v))
            convert(v, depth=depth + 1)
            if isinstance(v, omegaconf.dictconfig.DictConfig):
                dico[i] = OmegaConf.to_container(v)


def update_model(name, url, working_dir, new_entity, new_wandb_project_name):
    # Clear the working directory
    for file in os.listdir(working_dir):
        os.remove(osp.join(working_dir, file))

    os.chdir(working_dir)

    checkpoint_file = url.split("/")[-1]
    run_id = url.split("/")[-2]
    project = url.split("/")[-3]
    entity = url.split("/")[-4]

    logging.info("Downloading the old run files")
    api = wandb.Api()
    old_run = api.run(f"{entity}/{project}/runs/{run_id}")

    for file in old_run.files():
        file.download()

    logging.info("Converting checkpoint file")
    old_model_path = checkpoint_file
    dict_model = torch.load(old_model_path)
    convert(dict_model)
    torch.save(dict_model, checkpoint_file)

    with open("old_url.txt", "w") as f:
        f.write(f"https://wandb.ai/{entity}/{project}/runs/{run_id}")
    logging.info("Creating a new wandb run")
    wandb_args = {}
    wandb_args["project"] = new_wandb_project_name
    wandb_args["name"] = name
    run = wandb.init(**wandb_args)
    new_run_id = run.id
    wandb.finish()

    shutil.rmtree("wandb")

    run = api.run(f"{new_entity}/{new_wandb_project_name}/runs/{new_run_id}")

    # del run.summary["_wandb"]
    for key, value in old_run.summary.items():
        run.summary[key] = value
    run.summary.update()

    # Delete all files
    files = run.files()
    for file in files:
        if file.name != "wandb-summary.json":
            file.delete()

    logging.info("Uploading to new wandb run")
    for file in os.listdir(working_dir):
        run.upload_file(file)
    logging.info("Conversion finished")


def main():
    assert omegaconf.__version__ <= "1.4.1"

    url_list = models()
    for name, url in url_list:
        logging.info(f"Processing Model: {name}, URL: {url}")
        working_dir = "/home/leo/kios/torch-points3d/dev_stuff/test"
        new_entity = "leo_stan"
        new_wandb_project_name = "tp3d_public"
        update_model(name, url, working_dir, new_entity, new_wandb_project_name)


if __name__ == "__main__":
    main()
