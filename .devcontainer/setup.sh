set -eu

apt-get update
apt-get install -y --fix-missing --no-install-recommends libgl1-mesa-glx xvfb curl
curl -sL https://deb.nodesource.com/setup_15.x | bash -
apt-get update
apt-get install -y --fix-missing --no-install-recommends nodejs
apt-get clean &&rm -rf /var/lib/apt/lists/*

pip3 install pylint autopep8 flake8 pre-commit black 'jupyterlab<=2.0' pyvista panel
jupyter labextension install @pyviz/jupyterlab_pyviz
rm -rf /root/.cache
