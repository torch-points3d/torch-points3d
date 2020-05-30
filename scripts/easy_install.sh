#!/bin/bash

# Need to have called poetry shell before starting this processy

CUDA="$1"

sudo ls # Force sudo
curl -sSL https://bootstrap.pypa.io/get-pip.py | python
pip install --upgrade pip
pip cache purge
pip install --no-cache-dir poetry
pip install --no-cache-dir torch==1.4.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-cache-dir torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-cache-dir torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install --no-cache-dir torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install --no-cache-dirtorch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install --no-cache-dir torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-points3d
sudo pip uninstall protobuf
sudo pip uninstall google
sudo pip install --no-cache-dir google
sudo pip install --no-cache-dir protobuf
