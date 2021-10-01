set -eu

PIP=/venv/bin/pip3
PYTHON=/venv/bin/python

apt-get update
apt-get install -y --fix-missing --no-install-recommends libgl1-mesa-glx xvfb curl
curl -sL https://deb.nodesource.com/setup_15.x | bash -
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg |  gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/nul
apt-get update
apt-get install -y --fix-missing --no-install-recommends nodejs gh apt libusb-1.0-0
apt-get clean &&rm -rf /var/lib/apt/lists/*

curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | $PYTHON -

$PIP install pylint autopep8 flake8 pre-commit black 'jupyterlab<=2.0' pyvista panel
/venv/bin/jupyter labextension install @pyviz/jupyterlab_pyviz
rm -rf /root/.cache
