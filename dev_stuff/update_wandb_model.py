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

def download_file(url, out_file):
    # If the file exists, remove it
    if os.path.exists(out_file):
        os.remove(out_file)

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    urllib.request.urlretrieve(url, out_file)


def parse_args():

    parser = argparse.ArgumentParser("a simple script to convert omegaconf file to dict, you need omegaconf v 1.4.1 in order to convert files")
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
                convert(v, depth=depth+1)
                if isinstance(v, omegaconf.dictconfig.DictConfig):
                    dico[k] = OmegaConf.to_container(v)

    elif isinstance(dico, list):
        for i, v in enumerate(dico):
            # print(depth * " ", i, type(v))
            convert(v, depth=depth+1)
            if isinstance(v, omegaconf.dictconfig.DictConfig):
                dico[i] = OmegaConf.to_container(v)


def main():
    assert omegaconf.__version__ <= "1.4.1"
    # args = parse_args()
    # model_name = "pointnet2_largemsg-s3dis-2"
    url = "https://api.wandb.ai/files/loicland/benchmark-torch-points-3d-s3dis/1e1p0csk/pointnet2_largemsg.pt"
    working_dir = "/home/leo/kios/torch-points3d/dev_stuff/test"
    new_wandb_project_name = "test_project"

    # Clear the working directory
    for file in os.listdir(working_dir):
        os.remove(osp.join(working_dir, file))

    os.chdir(working_dir)

    model_name = url.split("/")[-1]
    run_id = url.split("/")[-2]
    project = url.split("/")[-3]
    entity = url.split("/")[-4]

    logging.info("Downloading the old run files")
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/runs/{run_id}")

    for file in run.files():
        file.download()

    logging.info("Converting checkpoint file")
    old_model_path = model_name
    dict_model = torch.load(old_model_path)
    convert(dict_model)
    torch.save(dict_model, model_name)

    with open("old_url.txt", "w") as f:
        f.write(url)

    logging.info("Creating a new wandb run")
    wandb_args = {}
    wandb_args["project"] = new_wandb_project_name
    run = wandb.init(**wandb_args)
    new_run_id = run.id
    wandb.finish()

    shutil.rmtree("wandb")

    run = api.run(f"leo_stan/{new_wandb_project_name}/runs/{new_run_id}")

    # Delete all files
    files = run.files()
    for file in files:
        file.delete()

    logging.info("Uploading to new wandb run")
    for file in os.listdir(working_dir):
        print(file)
        run.upload_file(file)
    logging.info("Conversion finished")


if __name__ == "__main__":
    main()

