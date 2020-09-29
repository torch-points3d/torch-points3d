:github_url: https://github.com/nicolas-chaulet/torch-points3d

Advanced
==========

Configuration
----------------

Overview
^^^^^^^^^^

We have chosen `Facebook Hydra library <https://hydra.cc/docs/intro>`_  as out core tool for managing the configuration of our experiments. It provides a nice and scalable interface to defining models and datasets. We encourage our users to take a look at their documentation and get a basic understanding of its core functionalities.
As per their website 

..

   "Hydra is a framework for elegantly configuring complex applications"


Configuration architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All configurations leave in the `conf <https://github.com/nicolas-chaulet/torch-points3d/tree/master/conf>`_ folder and it is organised as follow:

.. code-block:: bash

   .
   ├── config.yaml     # main config file for training
   ├── data            # contains all configurations related to datasets
   ├── debugging       # configs that can be used for debugging purposes
   ├── eval.yaml       # Main config for running a full evaluation on a given dataset
   ├── hydra           # hydra specific configs
   ├── lr_scheduler    # learning rate schedulers
   ├── models          # Architectures of the models
   ├── sota.yaml       # SOTA scores
   ├── training        # Training specific parameters
   └── visualization   # Parameters for saving visualisation artefact

Understanding config.yaml
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``config.yaml`` is the config file that governs the behaviour of your trainings. It gathers multiple configurations into one, and it is organised as follow:

.. code-block:: yaml

   defaults:
     - task: ??? # Task performed (segmentation, classification etc...)
       optional: True
     - model_type: ??? # Type of model to use, e.g. pointnet2, rsconv etc...
       optional: True
     - dataset: ???
       optional: True

     - visualization: default
     - lr_scheduler: multi_step
     - training: default
     - eval

     - debugging: default.yaml
     - models: ${defaults.0.task}/${defaults.1.model_type}
     - data: ${defaults.0.task}/${defaults.2.dataset}
     - sota # Contains current SOTA results on different datasets (extracted from papers !).
     - hydra/job_logging: custom

   model_name: ??? # Name of the specific model to load

   selection_stage: ""
   pretty_print: False

Hydra is expecting the followings arguments from the command line:


* task
* model_type
* dataset
* model_name

The provided ``task`` and ``dataset`` will be used to load the configuration for the dataset at ``conf/data/{task}/{dataset}.yaml`` while the ``model_type`` argument will be used to load the model config at ``conf/models/{task}/{model_type}.yaml``.
Finally ``model_name`` is used to pull the appropriate model from the model configuration file. 

Training arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../conf/training/default.yaml
   :language: yaml


* ``precompute_multi_scale``: Computes spatial queries such as grid sampling and neighbour search on cpu for faster. Currently this is only supported for KPConv.

Eval arguments
^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../conf/eval.yaml
   :language: yaml



Data formats for point cloud
------------------------------

While developing this project, we discovered there are several ways to implement a convolution.

* "DENSE"
* "PARTIAL_DENSE"
* "MESSAGE_PASSING"
* "SPARSE"

Dense
^^^^^^

This format is very similar to what you would be used to with images, during the assembling of a batch the B tensors of shape (num_points, feat_dim) will be concatenated on a new dimension
[(num_points, feat_dim), ..., (num_points, feat_dim)] -> (B, num_points, feat_dim).

This format forces each sample to have exactly the same number of points.

Advantages


* The format is dense and therefore aggregation operation are fast

Drawbacks


* Handling variability in the number of neighbours happens through padding which is not very efficient
* Each sample needs to have the same number of points, as a consequence points are duplicated or removed from a sample during the data loading phase using a FixedPoints transform


Sparse formats
^^^^^^^^^^^^^^^^

The second family of convolution format is based on a sparse data format meaning that each sample can have a variable number of points and the collate function handles the complexity behind the scene.
For those intersted in learning more about it `Batch.from_data_list <https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/batch.html#Batch>`_

.. image:: ../imgs/pyg_batch.PNG
   :target: ../imgs/pyg_batch.PNG
   :alt: Screenshot

Given ``N`` tensors with their own ``num_points_{i}``\ , the collate function does:

.. code-block::

    [(num_points_1, feat_dim), ..., (num_points_n, feat_dim)]
        -> (num_points_1 + ... + num_points_n, feat_dim)

It also creates an associated ``batch tensor`` of size ``(num_points_1 + ... + num_points_n)`` with indices of the corresponding batch.

.. note:: **Example**

    * A with shape (2, 2)
    * B with shape (3, 2)

    ``C = Batch.from_data_list([A, B])``

    C is a tensor of shape ``(5, 2)`` and its associated batch will contain ``[0, 0, 1, 1, 1]``


PARTIAL_DENSE ConvType format
""""""""""""""""""""""""""""""


This format is used by KPConv original implementation.

Same as dense format, it forces each point to have the same number of neighbors.
It is why we called it partially dense.


MESSAGE_PASSING ConvType Format
""""""""""""""""""""""""""""""""

This ConvType is Pytorch Geometric base format.
Using `Message Passing <https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/message_passing.html#MessagePassing>`_ API class, it deploys the graph created by ``neighbour finder`` using internally the ``torch.index_select`` operator.

Therefore, the ``[PointNet++]`` internal convolution looks like that.

.. code-block:: python

   import torch
   from torch_geometric.nn.conv import MessagePassing
   from torch_geometric.utils import remove_self_loops, add_self_loops

   from ..inits import reset

   class PointConv(MessagePassing):
       r"""The PointNet set layer from the `"PointNet: Deep Learning on Point Sets
       for 3D Classification and Segmentation"
       <https://arxiv.org/abs/1612.00593>`_ and `"PointNet++: Deep Hierarchical
       Feature Learning on Point Sets in a Metric Space"
       <https://arxiv.org/abs/1706.02413>`_ papers
       """

       def __init__(self, local_nn=None, global_nn=None, **kwargs):
           super(PointConv, self).__init__(aggr='max', **kwargs)

           self.local_nn = local_nn
           self.global_nn = global_nn

           self.reset_parameters()

       def reset_parameters(self):
           reset(self.local_nn)
           reset(self.global_nn)


       def forward(self, x, pos, edge_index):
           r"""
           Args:
               x (Tensor): The node feature matrix. Allowed to be :obj:`None`.
               pos (Tensor or tuple): The node position matrix. Either given as
                   tensor for use in general message passing or as tuple for use
                   in message passing in bipartite graphs.
               edge_index (LongTensor): The edge indices.
           """
           if torch.is_tensor(pos):  # Add self-loops for symmetric adjacencies.
               edge_index, _ = remove_self_loops(edge_index)
               edge_index, _ = add_self_loops(edge_index, num_nodes=pos.size(0))

           return self.propagate(edge_index, x=x, pos=pos)


       def message(self, x_j, pos_i, pos_j):
           msg = pos_j - pos_i
           if x_j is not None:
               msg = torch.cat([x_j, msg], dim=1)
           if self.local_nn is not None:
               msg = self.local_nn(msg)
           return msg

       def update(self, aggr_out):
           if self.global_nn is not None:
               aggr_out = self.global_nn(aggr_out)
           return aggr_out

       def __repr__(self):
           return '{}(local_nn={}, global_nn={})'.format(
               self.__class__.__name__, self.local_nn, self.global_nn)


SPARSE ConvType Format
"""""""""""""""""""""""

The sparse conv type is used by project like `SparseConv <https://github.com/facebookresearch/SparseConvNet>`_ or `Minkowski Engine <https://github.com/StanfordVL/MinkowskiEngine>`_,
therefore, the points have to be converted into indices living within a grid.



Backbone Architectures
------------------------

Several unet could be built using different convolution or blocks.
However, the final model will still be a UNet.

In the ``base_architectures`` folder, we intend to provide base architecture builder which could be used across tasks and datasets.

We provide two UNet implementations:


* UnetBasedModel
* UnwrappedUnetBasedModel

The main difference between them if ``UnetBasedModel`` implements the forward function and ``UnwrappedUnetBasedModel`` doesn't.

UnetBasedModel
^^^^^^^^^^^^^^^

.. code-block:: python

    def forward(self, data):
        if self.innermost:
            data_out = self.inner(data)
            data = (data_out, data)
            return self.up(data)
        else:
            data_out = self.down(data)
            data_out2 = self.submodule(data_out)
            data = (data_out2, data)
            return self.up(data)

The UNet will be built recursively from the middle using the ``UnetSkipConnectionBlock`` class.

**UnetSkipConnectionBlock**
.. code-block::
    
    Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    -- downsampling -- |submodule| -- upsampling --|

UnwrappedUnetBasedModel
^^^^^^^^^^^^^^^^^^^^^^^^

The ``UnwrappedUnetBasedModel`` will create the model based on the configuration and add the created layers within the followings ``ModuleList``

.. code-block:: python

     self.down_modules = nn.ModuleList()
     self.inner_modules = nn.ModuleList()
     self.up_modules = nn.ModuleList()


Datasets
---------


Segmentation
^^^^^^^^^^^^^

Preprocessed S3DIS
"""""""""""""""""""

We support a couple of flavours or `S3DIS <http://buildingparser.stanford.edu>`_. The dataset used for ``S3DIS1x1`` is coming from 
https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/s3dis.html. 

It is a preprocessed version of the original data where each sample is a 1mx1m extraction of the original data. It was initially used in PointNet. 


Raw S3DIS
"""""""""""""""""""

The dataset used for `S3DIS <http://buildingparser.stanford.edu>`_ is the original dataset without any pre-processing applied.
Here is the `area_1 <http://buildingparser.stanford.edu/rendered/raw_examples/Area%201.ply.html>`_ if you want to visualize it.
We provide some data transform for combining each area back together and split the dataset into digestible chunks. Please refer to `code base <https://github.com/nicolas-chaulet/torch-points3d/blob/master/torch_points3d/datasets/segmentation/s3dis.py>`_ and associated configuration file for more details:

.. literalinclude:: ../../conf/data/segmentation/s3disfused.yaml
   :language: yaml


Shapenet
"""""""""""""""""""

`Shapenet <https://www.shapenet.org/>`_ is a simple dataset that allows quick prototyping for segmentation models. 
When used in single class mode, for part segmentation on airplanes for example, it is a good way to figure out if your implementation is correct.

.. image:: ../imgs/shapenet.png
   :target: ../imgs/shapenet.png
   :alt: Screenshot



Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

ModelNet
"""""""""

The dataset used for ``ModelNet`` comes in two format:


* ModelNet10
* ModelNet40
  Their website is here https://modelnet.cs.princeton.edu/.


Registration
^^^^^^^^^^^^^^^^^^^^^^^^^^

3D Match
"""""""""

http://3dmatch.cs.princeton.edu/


IRALab Benchmark
"""""""""""""""""""

https://arxiv.org/abs/2003.12841 composed of data from:

* the ETH datasets (https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration)
* the Canadian Planetary Emulation Terrain 3D Mapping datasets (http://asrl.utias.utoronto.ca/datasets/3dmap/index.html)
* the TUM Vision Groud RGBD datasets (https://vision.in.tum.de/data/datasets/rgbd-dataset)
* the KAIST Urban datasets (https://irap.kaist.ac.kr/dataset)


Model checkpoint
------------------

Model Saving
^^^^^^^^^^^^^^^^^^^^

Our custom ``Checkpoint`` class keeps track of the models for ``every metric``\ , the stats for ``"train", "test", "val"``\ , ``optimizer`` and ``its learning params``.

.. code-block:: python

           self._objects = {}
           self._objects["models"] = {}
           self._objects["stats"] = {"train": [], "test": [], "val": []}
           self._objects["optimizer"] = None
           self._objects["lr_params"] = None

Model Loading
^^^^^^^^^^^^^^^^^^^^

In training.yaml and eval.yaml, you can find the followings parameters:


* weight_name
* checkpoint_dir
* resume

As the model is saved for every metric + the latest epoch.
It is possible by loading any of them using ``weight_name``.

Example: ``weight_name: "miou"``

If the checkpoint contains weight with the key "miou", it will set the model state to them. If not, it will try the latest if it exists. If None are found, the model will be randonmly initialized.


Adding a new metric
^^^^^^^^^^^^^^^^^^^^

Within the file ``torch_points3d/metrics/model_checkpoint.py``\ ,
It contains a mapping dictionnary between a sub ``metric_name`` and an ``optimization function``.

Currently, we support the following metrics.

.. code-block:: python

   DEFAULT_METRICS_FUNC = {
       "iou": max,
       "acc": max,
       "loss": min,
       "mer": min,
   }  # Those map subsentences to their optimization functions



Visualization
----------------

.. raw:: html

   <h6> The associated visualization </h6>


The framework currently support both `wandb <https://www.wandb.com/>`_ and `tensorboard <https://www.tensorflow.org/tensorboard>`_

.. code-block:: yaml

   # parameters for Weights and Biases
   wandb:
       project: benchmarking
       log: False

   # parameters for TensorBoard Visualization
   tensorboard:
       log: True

Custom logging
---------------
We use a custom hydra logging message which you can find within ``conf/hydra/job_logging/custom.yaml``

.. literalinclude:: ../../conf/hydra/job_logging/custom.yaml
   :language: yaml