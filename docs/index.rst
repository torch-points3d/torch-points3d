.. Torch Points 3D documentation master file, created by
   sphinx-quickstart on Wed Mar 18 08:19:48 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/nicolas-chaulet/torch-points3d

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. raw:: html

   <img src="https://raw.githubusercontent.com/nicolas-chaulet/torch-points3d/master/docs/logo.png" style="width: 40%; display: block; margin-left: auto; margin-right: auto;"/>
   <p/>


**Torch Points 3D** is a framework for developing and testing common
deep learning models to solve tasks related to unstructured 3D spatial data
i.e. Point Clouds. The framework currently integrates some of the best published
architectures and it  integrates the most common public datasests for ease of
reproducibility. It heavily relies on `Pytorch Geometric <https://github.com/rusty1s/pytorch_geometric>`_ and `Facebook Hydra library <https://hydra.cc/docs/intro>`_ thanks for the great work!

We aim to build a tool which can be used for benchmarking SOTA models, while also allowing practitioners to efficiently pursue research into point cloud analysis,  with the end-goal of building models which can be applied to real-life applications.


.. image:: imgs/Dashboard_demo.gif
   :target: imgs/Dashboard_demo.gif
   :alt: dashboard

Install with pip
-----------------
You can easily install Torch Points3D with ``pip``

.. code-block:: bash

   pip install torch
   pip install torch-points3d
   
but first make sure that the following dependencies are met

- CUDA 10 or higher (if you want GPU version)
- Python 3.6 or higher + headers (python-dev)
- PyTorch 1.7 or higher
- MinkowskiEngine (optional) see `here <https://github.com/nicolas-chaulet/torch-points3d#minkowski-engine>`_ for installation instructions





Core features
---------------


* **Task** driven implementation with dynamic model and dataset resolution from arguments.
* **Core** implementation of common components for point cloud deep learning - greatly simplifying the creation of new models:

  * **Core Architectures** - Unet
  * **Core Modules** - Residual Block, Down-sampling and Up-sampling convolutions
  * **Core Transforms** - Rotation, Scaling, Jitter
  * **Core Sampling** - FPS, Random Sampling,  Grid Sampling
  * **Core Neighbour Finder** - Radius Search, KNN

*
  4 **Base Convolution** base classes to simplify the implementation of new convolutions. Each base class supports a different data format (B = number of batches, C = number of features):


  * **DENSE** (B, num_points, C)
  * **PARTIAL DENSE** (B * num_points, C)
  * **MESSAGE PASSING** (B * num_points, C)
  * **SPARSE** (B * num_points, C)

*
  Models can be completely specified using a YAML file, greatly easing reproducability.

* Several visualiation tools **(tensorboard, wandb)** and **dynamic metric-based model checkpointing** , which is easily customizable.
* **Dynamic customized placeholder resolution** for smart model definition.

Supported models
------------------

The following models have been tested and validated:


* `Relation-Shape Convolutional (RSConv) Neural Network for Point Cloud Analysis <https://arxiv.org/abs/1904.07601>`_
* `KPConv: Flexible and Deformable Convolution for Point Clouds <https://arxiv.org/abs/1904.08889>`_
* `PointCNN: Convolution On X-Transformed Points <https://arxiv.org/abs/1801.07791>`_
* `PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space <https://arxiv.org/abs/1706.02413>`_
* `4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks <https://arxiv.org/abs/1904.08755>`_ 
* `Deep Hough Voting for 3D Object Detection in Point Clouds <https://arxiv.org/abs/1904.09664>`_

We are actively working on adding the following ones to the framework:

* `RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds  <https://arxiv.org/pdf/1911.11236.pdf>`_ - implemented but not completely tested

and much more to come ...

Supported tasks
---------------

* Segmentation
* Registration
* Classification
* Object detection

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Developer guide
   :hidden:

   src/gettingstarted
   src/tutorials
   src/advanced

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: API
   :hidden:

   src/api/models
   src/api/datasets
   src/api/transforms
   src/api/filters
