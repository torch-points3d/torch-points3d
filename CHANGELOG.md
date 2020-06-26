# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed

- evaluation output folder is now a subfolder of the checkpoint it uses
- saves model checkpoints to wandb
- GridSampling3D now creates a new attribute `coords` that stores the non quantized position when the transform is called in `quantize` mode

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
