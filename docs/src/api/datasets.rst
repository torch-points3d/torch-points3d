:github_url: https://github.com/nicolas-chaulet/torch-points3d

Datasets
========

Below is a list of the datasets we support as part of the framework. They all inherit from 
`Pytorch Geometric dataset <https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Dataset>`_ 
and they can be accessed either as raw datasets or wrapped into a 
`base class <https://github.com/nicolas-chaulet/torch-points3d/blob/afbab238f9c0e9d33651fa92a1b186664fdc9282/torch_points3d/datasets/base_dataset.py#L22>`_ that builds test, train and validations data loaders for you. 
This base class also provides a helper functions for pre-computing neighboors and point cloud sampling at data loading time.

ShapeNet
---------

Raw dataset
^^^^^^^^^^^
.. autoclass:: torch_points3d.datasets.segmentation.ShapeNet

Wrapped dataset
^^^^^^^^^^^^^^^
.. autoclass:: torch_points3d.datasets.segmentation.ShapeNetDataset


S3DIS
-----

Raw dataset
^^^^^^^^^^^
.. autoclass:: torch_points3d.datasets.segmentation.S3DISOriginalFused

.. autoclass:: torch_points3d.datasets.segmentation.S3DISSphere

Wrapped dataset
^^^^^^^^^^^^^^^
.. autoclass:: torch_points3d.datasets.segmentation.S3DIS1x1Dataset

.. autoclass:: torch_points3d.datasets.segmentation.S3DISFusedDataset


Scannet
-------

Raw dataset
^^^^^^^^^^^
.. autoclass:: torch_points3d.datasets.segmentation.Scannet

Wrapped dataset
^^^^^^^^^^^^^^^
.. autoclass:: torch_points3d.datasets.segmentation.ScannetDataset