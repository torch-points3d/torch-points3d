FROM ubuntu:bionic

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --fix-missing --no-install-recommends\
    libffi-dev libssl-dev build-essential libopenblas-dev\
    python3-pip python3-dev python3-venv python3-setuptools\
    git iproute2 procps lsb-release \
    libsm6 libxext6 libxrender-dev \
    xvfb libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV ENV=/py3-torch-points3d
ENV PYTHON=$ENV/bin/python
ENV PIP=$ENV/bin/pip

RUN python3 -m pip install -U pip \
    && pip3 install virtualenv \
    && python3 -m venv $ENV \
    && $PYTHON -m pip install -U pip \
    && $PIP install setuptools>=41.0.0 \
    && $PIP install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && $PIP install git+git://github.com/StanfordVL/MinkowskiEngine.git#7087411269605b8cb5b9f61246d474f507ca3781 \
    && rm -rf /root/.cache

WORKDIR /app

COPY pyproject.toml pyproject.toml
COPY torch_points3d/__init__.py torch_points3d/__init__.py
COPY README.md README.md

RUN $PIP install . && rm -rf /root/.cache
