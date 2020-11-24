FROM ubuntu:bionic

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

COPY install_system.sh install_system.sh
RUN bash install_system.sh

COPY install_python.sh install_python.sh
RUN bash install_python.sh cpu

ARG MODEL=""
ENV WORKDIR=/dpb
ENV MODEL_PATH=$WORKDIR/$MODEL

WORKDIR $WORKDIR

COPY pyproject.toml pyproject.toml
COPY torch_points3d/__init__.py torch_points3d/__init__.py
COPY README.md README.md
RUN pip3 install . && rm -rf /root/.cache

COPY . .

# Setup location of model for forward inference
RUN sed -i "/checkpoint_dir:/c\checkpoint_dir: $WORKDIR" forward_scripts/conf/config.yaml
RUN model_name=$(echo "$MODEL" | cut -f 1 -d '.') && sed -i "/model_name:/c\model_name: $model_name" forward_scripts/conf/config.yaml
