FROM ubuntu:bionic

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --fix-missing --no-install-recommends\
    libffi-dev libssl-dev build-essential \
    python3-pip python3-dev python3-venv python3-setuptools\
    git iproute2 procps lsb-release \
    libsm6 libxext6 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip
RUN pip3 install setuptools poetry

COPY pyproject.toml .
COPY poetry.lock .

RUN POETRY_VIRTUALENVS_CREATE=false poetry install --no-dev && rm -rf /root/.cache

ARG MODEL=""
ENV WORKDIR=/dpb
ENV MODEL_PATH=$WORKDIR/$MODEL

WORKDIR $WORKDIR
COPY . .

# Setup location of model for forward inference
RUN sed -i "/checkpoint_dir:/c\checkpoint_dir: $WORKDIR" forward_scripts/conf/config.yaml
RUN model_name=$(echo "$MODEL" | cut -f 1 -d '.') && sed -i "/model_name:/c\model_name: $model_name" forward_scripts/conf/config.yaml
