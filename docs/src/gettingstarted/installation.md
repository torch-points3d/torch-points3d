
## Install Python 3.6 or higher

Start by installing Python  > 3.6. You can use pyenv by doing the following:

```bash
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
```

Add these three lines to your ```.bashrc```
```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Finaly you can install `python 3.6.10` by running the following command

```bash
pyenv install 3.6.10
``` 

## Install dependencies using poetry

Start by installing poetry:

```bash
pip install poetry
```

You can clone the repository and install all the required dependencies as follow:

```bash
git clone https://github.com/nicolas-chaulet/deeppointcloud-benchmarks.git
cd deeppointcloud-benchmarks
pyenv local 3.6.10
poetry install
```
You can check that the install has been successful by running

```bash
poetry shell
python -m unittest
```

## Installation within a virtual environment

We try to maintain a `requirements.txt` file for those who want to use plain old `pip`. Start by cloning the repo:

```bash
git clone https://github.com/nicolas-chaulet/deeppointcloud-benchmarks.git
cd deeppointcloud-benchmarks
```

We still recommend that you first create a virtual environment and activate it before installing the dependencies:

```bash
python3 -m virtualenv pcb
source pcb/bin/activate
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

You should now be able to run the tests successfully:

```bash
python -m unittest
```