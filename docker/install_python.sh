if [[ "$#" -ne 1 ]]
then
    echo "Usage: ./install_python.sh gpu"
    exit 1
fi

python3 -m pip install -U pip
pip3 install pylint autopep8 flake8 pre-commit black mypy  # Dev tools
pip3 install setuptools>=41.0.0
if [ $1 == "gpu" ]; then
    echo "Install GPU"
    pip3 install torch==1.6.0 torchvision==0.7.0
    pip3 install MinkowskiEngine==v0.4.3 --install-option="--force_cuda" --install-option="--cuda_home=/usr/local/cuda"
    export FROCE_CUDA=1 && pip3 install git+https://github.com/mit-han-lab/torchsparse.git
    pip3 install pycuda
else
    echo "Install CPU"
    pip3 install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    pip3 install MinkowskiEngine==v0.4.3
    pip3 install git+https://github.com/mit-han-lab/torchsparse.git
fi

rm -rf /root/.cache
