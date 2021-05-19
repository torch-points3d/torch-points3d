import argparse
import os
import os.path as osp
import torch
from omegaconf import OmegaConf
import omegaconf


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
                print(depth * " ", k, type(v))
                convert(v, depth=depth+1)
                if isinstance(v, omegaconf.dictconfig.DictConfig):
                    dico[k] = OmegaConf.to_container(v)

    elif isinstance(dico, list):
        for i, v in enumerate(dico):
            print(depth * " ", i, type(v))
            convert(v, depth=depth+1)
            if isinstance(v, omegaconf.dictconfig.DictConfig):
                dico[i] = OmegaConf.to_container(v)





if __name__ == "__main__":
    args = parse_args()

    assert omegaconf.__version__ <= "1.4.1"
    dict_model = torch.load(args.file)
    if (args.old):
        torch.save(dict_model, args.old)
    convert(dict_model)
    torch.save(dict_model, args.out)

    # print(omegaconf.OmegaConf.to_container)
