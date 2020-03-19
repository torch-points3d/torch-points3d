.. Deep Point Cloud Benchmark documentation master file, created by
   sphinx-quickstart on Wed Mar 18 08:19:48 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: "https://github.com/nicolas-chaulet/deeppointcloud-benchmarks"

Deep Point Cloud Benchmark
======================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. image:: logo.png
   :target: logo.png
   :alt: Screenshot


**Deep Point Cloud Benchmark** is a framework for developing and testing common 
deep learning models to solve tasks related to unstructured 3D spatial data 
i.e. Point Clouds. The framework currently integrates some of the best published 
architectures and it  integrates the most common public datasests for ease of 
reproducibility. It heavily relies on `Pytorch Geometric <https://github.com/rusty1s/pytorch_geometric>`_ and `Facebook Hydra library <https://hydra.cc/docs/intro>`_ thanks for the great work!

We aim to build a tool which can be used for benchmarking SOTA models, while also allowing practitioners to efficiently pursue research into point cloud analysis,  with the end-goal of building models which can be applied to real-life applications.


.. image:: imgs/Dashboard_demo.gif
   :target: imgs/Dashboard_demo.gif
   :alt: dashboard


Core features
---------------


* **Task** driven implementation with dynamic model and dataset resolution from arguments.
* **Core** implementation of common components for point cloud deep learning - greatly simplifying the creation of new models:

  * **Core Architectures** - Unet
  * **Core Modules** - Residual Block, Down-sampling and Up-sampling convolutions
  * **Core Transforms** - Grid Sampling, Rotation, Scaling
  * **Core Sampling** - FPS, Random Sampling
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

We are actively working on adding the following ones to the framework:

* `RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds  <https://arxiv.org/pdf/1911.11236.pdf>`_ - implemented but not completely tested
* `4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks <https://arxiv.org/abs/1904.08755>`_ - Actively being worked on

and much more to come ...

Supported tasks
---------------

* Segmentation
* Registration
* Classification [TO COME]
* Object detection [TO COME]


.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
    
   src/gettingstarted
   src/tutorials
   src/advanced

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API
   :hidden:
    
   src/api/transforms