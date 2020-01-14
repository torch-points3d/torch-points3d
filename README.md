![Project Logo](/docs/logo.png)

This is a framework for running common deep learning models for point cloud analysis tasks against classic benchmark. It heavily relies on pytorch geometric and hydra core.

**[Documentation](https://deeppointcloud-benchmarks.readthedocs.io/en/latest/)** | **[Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/resources.html)** | **[Facebook Hydra](https://hydra.cc/)**

The framework allows lean and yet complex model to be built with minimum effort and great reproducibility.

# COMPACT API
```yaml
  # PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space (https://arxiv.org/abs/1706.02413)
    pointnet2_original:
        type: pointnet2_dense
        down_conv:
            npoint: [1024, 256, 64, 16]
            radii: [[0.05, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.8]]
            nsamples: [[16, 32], [16, 32], [16, 32], [16, 32]]
            down_conv_nn:
                [
                    [[FEAT, 16, 16, 32], [FEAT, 32, 32, 64]],
                    [[32 + 64, 64, 64, 128], [32 + 64, 64, 96, 128]],
                    [[128 + 128, 128, 196, 256], [128 + 128, 128, 196, 256]],
                    [[256 + 256, 256, 256, 512], [256 + 256, 256, 384, 512]],
                ]
        up_conv:
            up_conv_nn:
                [
                    [512 + 512 + 256 + 256, 512, 512],
                    [512 + 128 + 128, 512, 512],
                    [512 + 64 + 32, 256, 256],
                    [256 + FEAT, 128, 128],
                ]
            skip: True
        mlp_cls:
            nn: [128, 128]
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
| [```pointnet2_original```](/benchmark/s3dis_fold5/Pointnet2_original.md)| 3,026,829 | 04:29 / 01:07 | 0.0512 | 85.26 | 45.58 | 73.11

## Shapenet part segmentation
The data reported below correspond to the part segmentation problem for Shapenet for all categories. The mean values reported are the mean of each per category metric.
| Model Name | Use Normals | Size | Speed Train / Test | Cross Entropy | OAcc | mIou | mAcc |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [```pointnet2_original```](/benchmark/shapenet/pointnet2_original.md)| Yes | 3,026,829 | 05:15 / 00:33 | 0.089 | 93.40 | 90.81 | 87.90

## Contributing
Contributions are welcome! The only asks are that you stick to the styling and that you add tests as you add more features!
For styling you can use [pre-commit hooks](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/) to help you:
```
pre-commit install
```
A sequence of checks will be run for you and you may have to add the fixed files again to the stahed files.

## Contributers
- [Thomas Chaton](https://github.com/tchaton)
- [Nicolas Chaulet](https://github.com/nicolas-chaulet)
- [Tristan Heywood](https://github.com/tristanheywood)
