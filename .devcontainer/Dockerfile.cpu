FROM principialabs/torch-points3d:latest-cpu

COPY .devcontainer/setup.sh setup.sh
RUN bash setup.sh
ENV PATH="${PATH}:$HOME/.poetry/bin"