set -eu

if [[ "$#" -ne 1 ]]
then
    echo "Usage: ./install_python.sh gpu"
    exit 1
fi

python3 -m pip install -U pip
pip3 install setuptools>=41.0.0
if [ $1 == "gpu" ]; then
    echo "Install GPU"
    pip3 install torch==1.8.1
    pip3 install MinkowskiEngine --install-option="--force_cuda" --install-option="--cuda_home=/usr/local/cuda"
    pip3 install git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0 -v
    pip3 install pycuda
else
    echo "Install CPU"
    pip3 install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
    pip3 install MinkowskiEngine
    pip3 install git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
fi
