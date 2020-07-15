FROM ubuntu:bionic

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --fix-missing --no-install-recommends\
    libffi-dev libssl-dev build-essential libopenblas-dev\
    python3-pip python3-dev python3-venv python3-setuptools\
    git iproute2 procps lsb-release \
    libsm6 libxext6 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip \
    && python3 -m pip install -U pip \
    && pip3 install setuptools>=41.0.0 \
    && pip3 install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install git+git://github.com/StanfordVL/MinkowskiEngine.git#v0.4.3 \
    && rm -rf /root/.cache

COPY pyproject.toml pyproject.toml
COPY torch_points3d/__init__.py torch_points3d/__init__.py
COPY README.md README.md

RUN pip3 install . && rm -rf /root/.cache

ARG MODEL=""
ENV WORKDIR=/dpb
ENV MODEL_PATH=$WORKDIR/$MODEL

WORKDIR $WORKDIR
COPY . .

# Setup location of model for forward inference
RUN sed -i "/checkpoint_dir:/c\checkpoint_dir: $WORKDIR" forward_scripts/conf/config.yaml
RUN model_name=$(echo "$MODEL" | cut -f 1 -d '.') && sed -i "/model_name:/c\model_name: $model_name" forward_scripts/conf/config.yaml
