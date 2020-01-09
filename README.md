![Project Logo](/docs/logo.png)

This is a framework for running common deep learning models for point cloud analysis tasks against classic benchmark. It heavily relies on pytorch geometric and hydra core.

**[Documentation](https://deeppointcloud-benchmarks.readthedocs.io/en/latest/)** | **[Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/resources.html)** | **[Facebook Hydra](https://hydra.cc/)**

The framework allows lean and yet complex model to be built with minimum effort and great reproducibility.

# COMPACT API
```yaml
  # PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space (https://arxiv.org/abs/1706.02413)
  pointnet2:
      type: pointnet2
      down_conv:
          module_name: SAModule
          ratios: [0.2, 0.25]
          radius: [0.2, 0.4]
          down_conv_nn: [[FEAT + 3, 64, 64, 128], [128 + 3, 128, 128, 256]]
      up_conv:
          module_name: FPModule
          up_conv_nn: [[1024 + 256, 256, 256], [256 + 128, 256, 128], [128 + FEAT, 128, 128, 128]]
          up_k: [1, 3, 3]   
          skip: True    
      innermost:
          module_name: GlobalBaseModule
          aggr: max
          nn: [256 + 3, 256, 512, 1024]
      mlp_cls: 
          nn: [128, 128, 128, 128, 128]
          dropout: 0.5
```

## Getting started
You will first need to install [poetry](https://poetry.eustace.io/) in order to setup a virtual environments and install the relevant packages, then run
```
poetry install
```
This will install all required dependencies in a new virtual environment.

## Train pointnet++ on Segmentation task for dataset shapenet
```
poetry run python train.py experiment.name=pointnet2 experiment.data=shapenet
```
And you should see something like that

![logging](/docs/imgs/logging.png)

# Benchmark
## S3DIS


| Model Name | Size | Speed Train / Test | Cross Entropy | OAcc | mIou | mAcc |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [```pointnet2```](/benchmark/s3dis_fold5/Pointnet2_original.md)| 3,026,829 | 04:29 / 01:07 | 0.0512 | 85.26 | 45.58 | 73.11 

## Contributing
We use [autopep8](https://github.com/hhatto/autopep8) for formating with the following options:
`--max-line-length 120 --ignore E402,E226,E24,W50,W690`

## Contributers
- [Thomas Chaton](https://github.com/tchaton)
- [Nicolas Chaulet](https://github.com/nicolas-chaulet)
- [Tristan Heywood](https://github.com/tristanheywood)
