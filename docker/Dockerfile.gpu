FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.8-venv  python3.8-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

ENV FORCE_CUDA 1
ENV TORCH_CUDA_ARCH_LIST "3.5 5.2 6.0 6.1 7.0+PTX"

COPY docker/install_system.sh install_system.sh
RUN bash install_system.sh

COPY docker/install_python.sh install_python.sh
RUN bash install_python.sh gpu && rm -rf /root/.cache

ENV WORKDIR=/tp3d
WORKDIR $WORKDIR

COPY pyproject.toml pyproject.toml
COPY torch_points3d/__init__.py torch_points3d/__init__.py
COPY README.md README.md

RUN pip3 install . && rm -rf /root/.cache

COPY . .
