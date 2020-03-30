#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------

FROM ubuntu:bionic

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# This Dockerfile adds a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container user's GID/UIDs
# will be updated to match your local UID/GID (when using the dockerFile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Uncomment the following COPY line and the corresponding lines in the `RUN` command if you wish to
# include your requirements in the image itself. It is suggested that you only do this if your
# requirements rarely (if ever) change.

RUN apt-get update \
    && apt-get install -y --fix-missing --no-install-recommends\
    libffi-dev libssl-dev build-essential libopenblas-dev\
    python3-pip python3-dev python3-venv python3-setuptools\
    git iproute2 procps lsb-release \
    libsm6 libxext6 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip
RUN pip3 install setuptools poetry

COPY pyproject.toml .
COPY poetry.lock .

RUN poetry install --no-root
RUN poetry env info --path > /python_path
RUN cat /python_path

RUN git clone https://github.com/StanfordVL/MinkowskiEngine.git /tmp/ME \
    && cd /tmp/ME \
    && export VENV=`cat /python_path` \
    && $VENV/bin/python setup.py install --cpu_only
