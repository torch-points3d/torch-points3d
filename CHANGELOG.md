# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unrealeased

### Changed

- Migrated to Hydra 1.0 and OmegaConf 2.1 **BREAKING any checkpoint created prior to that, in particular the model zoo**

### Added

- Mixed precision training support for SparseConv3D models with torchsparse backend (Requires torchsparse >= 1.3.0)

## 1.3.0

### Added

- MS-SVConv: https://arxiv.org/abs/2103.14533 (thanks @humanpose1)
- added new data generations techniques for the self-supervised learning (PeriodicSampling, IrregularSampling EllipsoidCrop) (thanks @humanpose1)
- More ETH benchmark dataset (thanks @humanpose1)

### Changed

- Minkowski 0.5 support

### Bug fixes

- Fix bug in data loader https://github.com/nicolas-chaulet/torch-points3d/issues/443 thanks @JloveU
- Fix bug in base unet that created problems when loading pointnet++ model checkpoint

## 1.2.0

### Added

- Support for the IRALab benchmark (https://arxiv.org/abs/2003.12841), with data from the ETH, Canadian Planetary, Kaist and TUM datasets. (thanks @simone-fontana)
- Added Kitti for semantic segmentation and registration (first outdoor dataset for semantic seg)
- Possibility to load pretrained models by adding the path in the confs for finetuning.
- Lottery transform to use randomly selected transforms for data augmentation
- Batch size campling function to ensure that batches don't get too large
- [TorchSparse](https://github.com/mit-han-lab/torchsparse) backend for sparse convolutions
- Possibility to build sparse convolution networks with Minkowski Engine or TorchSparse
- [PVCNN](https://arxiv.org/abs/1907.03739) model for semantic segmentation (thanks @CCInc)

### Bug fix

- Dataset configurations are saved in the checkpoints so that models can be created without requiring the actual dataset
- Trainer was giving a warning for models that could not be re created when they actually could
- BatchNorm1d fix (thanks @Wundersam)
- Fix process hanging when processing scannet with multiprocessing (thanks @zetyquickly)
- wandb does not log the weights when set in private mode (thanks @jamesjiro)
- Fixed VoteNet loss definitions and data augmentation parameters (got up to 59.2% mAP25)

### Changed

- More general API for Minkowski with support for Bottleneck blocks and Squeeze and excite.
- Docker images tags on dockerhub are now `latest-gpu` and `latest-cpu` for the latest CPU adn GPU images.

### Removed

- Removed VoteNet from the API because it was not up to date. You can still use the models defined [there](https://github.com/nicolas-chaulet/torch-points3d/tree/master/torch_points3d/models/object_detection)

## 1.1.1

### Added

- Teaser support for registration
- Examples for using pretrained registration models
- Pointnet2 forward examples for classification, segmentation
- S3DIS automatic download and panoptic support and cylinder sampling

### Changed

- Moved to PyTorch 1.6 as officialy supported PyTorch version

### Bug fix

- Add `context = ssl._create_unverified_context()`, `data = urllib.request.urlopen(url, context=context)` within `download_ulr`, so ModelNet and ShapeNet can download.

## 1.1.0

### Added

- Support scannet test dataset and automatic generation of submission files using the eval.py script
- Full res predictions on Scannet with voting
- VoteNet model and losses
- Tracker for object detection
- Models can specify which attributes they need from the data in order to forward and train properly
- Full res predictions on ShapeNet with voting
- Trainer class to handle train / eval
- Add testing for Trainer:
  - Segmentation: PointNet2 on cap ShapeNet
  - Segmentation: KPConv on scannetV2
  - Object Detection: VoteNet on scannetV2
- Add VoteNet Paper / Backbones within API
- Windows support
- Weights are uploaded to wandb at the end of the run
- Added PointGroup https://arxiv.org/pdf/2007.01294.pdf
- Added PretrainedRegistry allowing model weight to be downloaded directly from wandb and DatasetMocking
- Added script for s3dis cross-validation [scripts/cv_s3dis.py]. 6 different pretrained models will be downloaded, evaluated on full resolution and confusion matrice will be summed to get all metrics.
- mAP tracker for Panoptic segmentation

### Changed

- evaluation output folder is now a subfolder of the checkpoint it uses
- saves model checkpoints to wandb
- GridSampling3D now creates a new attribute `coords` that stores the non quantized position when the transform is called in `quantize` mode
- cuda parameter can be given in command line to select the GPU to use
- Updated to pytorch geometric 1.6.0

### Bugfix

- LR secheduler resume is broken for update on batch number #328
- ElasticDistortion transform is now fully functional

### Removed

## 1.0.1

### Changed

- We now support the latest PyTorch
- Migration to the latest PyTorch Geometric and dependencies

### Bugfixes

- #273 (support python 3.7)

## 0.2.2

### Bugfix

- Pre transform is being overriden by the inference transform

## 0.2.1

### Added

- Customizable number of channels at the output of the API models
- API models expose output number of channels as a property
- Added Encoder to the API
- Sampled ModelNet dataset for point clouds
