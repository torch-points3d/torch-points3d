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
- Full res predictions on ShapeNet with voting

### Changed

- evaluation output folder is now a subfolder of the checkpoint it uses

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
