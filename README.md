![Project Logo](/docs/logo.png)

This is a framework for running common deep learning models for point cloud analysis tasks against classic benchmark. It heavily relies on pytorch geometric and hydra core.

# Benchmark
## S3DIS

[RSConv 2LD](/benchmark/s3dis_fold5/RSConv_2LD.md)

## Getting started
You will first need to install [poetry](https://poetry.eustace.io/) in order to setup a virtual environments and install the relevant packages, then run
```
poetry install
```
This will install all required dependencies in a new virtual environment.

## Benchmark pointnet++ on Segmentation task for shapenet (default dataset)
```
poetry run python train.py tested_model.name=pointnet2
```

## Documentation

To come :)

## Contributing
We use [autopep8](https://github.com/hhatto/autopep8) for formating with the following options:
`--max-line-length 120 --ignore E402,E226,E24,W50,W690`

## Contributers
- [Thomas Chaton](https://github.com/tchaton)
- [Nicolas Chaulet](https://github.com/nicolas-chaulet)
- [Tristan Heywood](https://github.com/tristanheywood)
