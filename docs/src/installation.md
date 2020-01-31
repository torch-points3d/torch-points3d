# Advised Installation Process


<h4> pyenv installer </h4>

pyenv.run redirects to the install script in this repository and the invocation above is equivalent to:

```
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
```

Add these three lines to your ```.bashrc```
```
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Download python 3.6.10
```
pyenv install 3.6.10
```

<h4> Install poetry </h4>

```
pip install poetry
```

<h4> Clone our repository </h4>

```
git clone https://github.com/nicolas-chaulet/deeppointcloud-benchmarks.git
```

<h4> Install project dependencies </h4>

Get within the project
```
cd deeppointcloud-benchmarks
```

Set python using pyenv local env
```
pyenv local 3.6.10
```

Install all dependencies
```
poetry update
```

```You are good to go !```

# Troubleshooting


<h4> Undefined symbol / Updating pytorch </h4>

When we update the version of pytorch that is used, the compiled packages need to be reinstalled, otherwise you will run into an error that looks like this:
```
... scatter_cpu.cpython-36m-x86_64-linux-gnu.so: undefined symbol: _ZN3c1012CUDATensorIdEv
```
This can happen for the following libraries:

* torch-points
* torch-scatter
* torch-cluster
* torch-sparse

An easy way to fix this is to run the following command with the virtualenv activated:

```
pip uninstall torch-scatter torch-sparse torch-cluster torch-points
rm -rf ~/.cache/pip
poetry install
```
