set -eu

apt-get update
apt-get install -y --fix-missing --no-install-recommends\
    libffi-dev libssl-dev build-essential libopenblas-dev libsparsehash-dev\
    python3-pip python3-dev python3-venv python3-setuptools\
    git iproute2 procps lsb-release \
    libsm6 libxext6 libxrender-dev ninja-build
apt-get clean
rm -rf /var/lib/apt/lists/*
