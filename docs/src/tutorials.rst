:github_url: https://github.com/nicolas-chaulet/torch-points3d

Tutorials
===========

Here you will learn how you can extend the framework to serve your needs, we will cover

.. contents::
    :local:
    :depth: 1

Create a new dataset
-----------------------

Let's add support for the version of `S3DIS <http://buildingparser.stanford.edu/dataset.html>`_ that **Pytorch Geometric** provides:
https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.S3DIS

We are going to go through the successive steps to do so:

.. contents::
    :local:
    :depth: 1


Let's go through those steps together and in order to go further, we strongly advice to read the `Creating Your Own Datasets <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html>`_ from **Pytorch Geometric**.

Create a dataset that the framework recognises
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The framework provides a base class for datasets that needs to be sub classed when you add your own.
We also follow the convention that the ``.py`` file that describes a dataset for segmentation will live in the ``torch_points3d/datasets/segmentation`` folder.
For another task such as classification it would go in ``torch_points3d/datasets/classification``.

Start by creating a new file ``torch_points3d/datasets/segmentation/s3dis.py`` with the class ``S3DISDataset``, it should inherit from ``BaseDataset``.

.. code-block:: python

    from torch_geometric.datasets import S3DIS

    from torch_points3d.datasets.base_dataset import BaseDataset
    from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

    class S3DISDataset(BaseDataset):
        def __init__(self, dataset_opt):
            super().__init__(dataset_opt)

            self.train_dataset = S3DIS(
                self._data_path,
                test_area=self.dataset_opt.fold,
                train=True,
                pre_transform=self.pre_transform,
                transform=self.train_transform,
            )
            self.test_dataset = S3DIS(
                self._data_path,
                test_area=self.dataset_opt.fold,
                train=False,
                pre_transform=self.pre_transform,
                transform=self.test_transform,
            )


        def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
            """Factory method for the tracker

            Arguments:
                wandb_log - Log using weight and biases
                tensorboard_log - Log using tensorboard
            Returns:
                [BaseTracker] -- tracker
            """
            return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

Let's explain the code more in details there.

.. code-block:: python

   class S3DISDataset(BaseDataset):
        def __init__(self, dataset_opt):
            super().__init__(dataset_opt)

This instantiates the parent class based on a given configuration ``dataset_opt`` (see :ref:`config_ref`) and this does few things for you:

* Sets the path to the data, by convention it will be ``dataset_opt.dataroot/s3dis/`` in our case (name of the class without Dataset)
* Extracts from the configuration the transforms that should be applied to your data before giving it to the model


Next comes the instantiation of the actual datasets that will be used for training and testing.

.. code-block:: python

        self.train_dataset = S3DIS(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=True,
            pre_transform=self.pre_transform,
            transform=self.train_transform,
        )
        self.test_dataset = S3DIS(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=False,
            pre_transform=self.pre_transform,
            transform=self.test_transform,
        )

You can see that we use the ``pre_transform``, ``test_transform`` and ``train_transform`` from the base class, they have been set based on the configuration
that you have provided. The base class will then use those datasets to create the dataloaders that will be used in the training script.

The final step is to associate a metric tracker to your dataset, in this case we will use a SegmentationTracker that tracks IoU metrics as well as accuracy, mean accuracy and loss.

.. code-block:: python

        def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
            """Factory method for the tracker

            Arguments:
                wandb_log - Log using weight and biases
                tensorboard_log - Log using tensorboard
            Returns:
                [BaseTracker] -- tracker
            """
            return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)


.. _config_ref:

Create a new configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's move to the next step, the definition of the configuration file that will control the behaviour of our dataset. The configuration file mainly controls the following things:

* Location of the data
* Transforms that will be applied to the data
* Python class that will be used for creating the actual python object used during training.

Let's create a ``conf/data/segmentation/s3disfused.yaml`` file with our own setting to setup the dataset

.. literalinclude:: ../../conf/data/segmentation/s3disfused.yaml
   :language: yaml

.. note::
 * ``task`` needs to be specified. Currently, the arguments provided by the command line are lost and therefore we need the extra information.
 * ``class`` needs to be specified. In that case, since we solve a classification task, the code will look for a class named ``S3DISDataset`` within the ``torch_points3d/datasets/segmentation/s3dis.py`` file.

For more details about the tracker please refer to the `source code <https://github.com/nicolas-chaulet/torch-points3d/blob/master/torch_points3d/metrics/segmentation_tracker.py>`_

Create a new model
--------------------

Let's add `PointNet++ <https://github.com/charlesq34/pointnet2>`_ model implemented within the "DENSE" format type to the project.
Model definitions are separated between the definition of the core "convolution" operation equivalent to a Conv2D on images (see :ref:`module_ref`) and the overall model that combines all those convolutions (see :ref:`model_ref`).


We are going to go through the successive steps to do so:

.. contents::
   :local:
   :depth: 1

.. _module_ref:

Create the basic modules
^^^^^^^^^^^^^^^^^^^^^^^^^


Let's create ``torch_points3d/modules/pointnet2/`` directory and ``dense.py`` file within.

.. note::
 Remember to create a ``__init__.py`` file within that directory that will contain the multiscale convolution proposed in pointnet++.

.. literalinclude:: ../../torch_points3d/modules/pointnet2/dense.py
   :language: python


Let's dig in.

.. code-block:: python

   class PointNetMSGDown(BaseDenseConvolutionDown):
       def __init__(
           ...
       ):
           super(PointNetMSGDown, self).__init__(
               DenseFPSSampler(num_to_sample=npoint), DenseRadiusNeighbourFinder(radii, nsample), **kwargs
           )


* The ``PointNetMSGDown`` inherit from ``BaseDenseConvolutionDown``:

  * ``BaseDenseConvolutionDown`` takes care of all the sampling and search logic for you.
  * Therefore, a ``sampler`` and a ``neighbour finder`` have to be provided.
  * Here, we provide ``DenseFPSSampler`` (furthest point sampling) and ``DenseRadiusNeighbourFinder`` (neighbour search within a given radius)

* The ``PointNetMSGDown`` class just needs to implement the ``conv`` method which implements the actual logic for deriving the features that will come out of this layer. Here the features of a given point are obtained by passing the neighbours of that point through an MLP.

.. _model_ref:

Assemble all the basic blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's create a new file ``/torch_points3d/models/segmentation/pointnet2.py`` with its associated class
``PointNet2_D``

.. code-block:: python

   import torch

   import torch.nn.functional as F
   from torch_geometric.data import Data
   import logging

   from torch_points3d.modules.pointnet2 import * # This part is extremely important. Always important the associated modules within your this file
   from torch_points3d.core.base_conv.dense import DenseFPModule
   from torch_points3d.models.base_architectures import UnetBasedModel

   log = logging.getLogger(__name__)

   class PointNet2_D(UnetBasedModel):
       def __init__(self, option, model_type, dataset, modules):
           """Initialize this model class.
           Parameters:
               opt -- training/test options
           A few things can be done here.
           - (required) call the initialization function of BaseModel
           - define loss function, visualization images, model names, and optimizers
           """
           UnetBasedModel.__init__(
               self, option, model_type, dataset, modules
           )  # call the initialization method of UnetBasedModel

           # Create the mlp to classify data
           nn = option.mlp_cls.nn
           self.dropout = option.mlp_cls.get("dropout")
           self.lin1 = torch.nn.Linear(nn[0], nn[1])
           self.lin2 = torch.nn.Linear(nn[2], nn[3])
           self.lin3 = torch.nn.Linear(nn[4], dataset.num_classes)

           self.loss_names = ["loss_seg"] # This will be used the automatically get loss_seg from self

       def set_input(self, data, device):
           """Unpack input data from the dataloader and perform necessary pre-processing steps.
           Parameters:
               input: a dictionary that contains the data itself and its metadata information.
           """
           data = data.to(device)
           self.input = data
           self.labels = data.y
           self.batch_idx = torch.arange(0, data.pos.shape[0]).view(-1, 1).repeat(1, data.pos.shape[1]).view(-1)

       def forward(self) -> Any:
           """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
           data = self.model(self.input)
           x = F.relu(self.lin1(data.x))
           x = F.dropout(x, p=self.dropout, training=self.training)
           x = self.lin2(x)
           x = F.dropout(x, p=self.dropout, training=self.training)
           x = self.lin3(x)
           self.output = F.log_softmax(x, dim=-1)
           return self.output

       def backward(self):
           """Calculate losses, gradients, and update network weights; called in every training iteration"""
           # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
           # calculate loss given the input and intermediate results
           self.loss_seg = F.nll_loss(self.output, self.labels) + self.get_internal_loss()

           self.loss_seg.backward()  # calculate gradients of network G w.r.t. loss_G

.. note::

    * Make sure that you import all the required modules
    * You need to inherit from ``BaseModel``. That class contains all the core logic that enables training (see `base_model.py <https://github.com/nicolas-chaulet/torch-points3d/blob/master/torch_points3d/models/base_model.py>`_ for more details)

Create a new configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We create a new file ``conf/models/segmentation/pointnet2.yaml``.
This file will contain all the **different versions** of pointnet++.

.. literalinclude:: ../../conf/models/segmentation/pointnet2.yaml
   :language: yaml
   :lines: 1,87-122

Here is ``PointNet++`` Multi-Scale original version by `Charles R. Qi <https://github.com/charlesq34>`_.

Let's dig in the definition.


**Required arguments**

*
  ``pointnet2_charlesmsg`` is model_name and should be provided from the command line in order to load this file configuration.

*
  ``architecture: pointnet2.PointNet2_D``. It indicates where to find the Model Logic.
  The framework backend will look for the file ``/torch_points3d/models/segmentation/pointnet2.py`` and the ``PointNet2_D`` class.

*
  ``conv_type: "DENSE"``


**"Optional" arguments**


When I say optional, I mean those parameters could be defined differently for your own model.
We don't want to force any particular configuration format however, the simpler is always better !

The format above is used across models that leverage our  `Unet architecture <https://arxiv.org/abs/1505.04597>`_ builder base class
`torch_points3d/models/base_architectures/unet.py <https://github.com/nicolas-chaulet/torch-points3d/blob/master/torch_points3d/models/base_architectures/unet.py>`_
with ``UnetBasedModel`` and ``UnwrappedUnetBasedModel``.
The following arguments are required by those classes:

* ``down_conv``: parameters of each down convolution layer
* ``innermost``: parameters of the innermost layer
* ``up_conv``: parameters of each up convolution layer

Those elements need to contain a ``module_name`` which will be used to create the associated Module.

Those Unet builder classes will do the followings:

* If provided a list, it will use the index to access the value
* If provided something else, it will broadcast the arguments to all convolutions.

**Understanding the model**

From the configuration written above, we can infer that

- The model has got two down convolutions, one inner module and three up convolutions
- Each down convolutions is a multiscale pointnet++ convolution implemented with the class ``PointNetMSGDown``
- The first down convolution uses the following parameters:

    - only 512 points are kept after this layer,
    - three scales with radii 0.1, 0.2 and 0.4 are used,
    - 32, 64 and 128 neighbours are kept for each scale
    - the multi layer perceptrons for each scale are of size: [FEAT+3, 32, 32, 64], [FEAT+3, 64, 64, 128] and [FEAT+3, 64, 96, 128] respectively
- The up convolution uses ``DenseFPModule`` and the first layer has got an MLP of size  [1024 + 256*2, 256, 256]
- The final classifier has got two layers and uses a dropout of 0.5

Another example with RSConcv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here is an example with the ``RSConv`` implementation in ``MESSAGE_TYPE ConvType``.

.. code-block:: python

   class RSConvDown(BaseConvolutionDown):
       def __init__(self, ratio=None, radius=None, local_nn=None, down_conv_nn=None, *args, **kwargs):
           super(RSConvDown, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)

           self._conv = Convolution(local_nn=local_nn, global_nn=down_conv_nn)

       def conv(self, x, pos, edge_index, batch):
           return self._conv(x, pos, edge_index)

We can see this convolution needs the followings arguments

.. code-block:: python

   ratio=None, radius=None, local_nn=None, down_conv_nn=None

Here is an extract from the model architecture config:

.. code-block:: yaml

   down_conv: # For the encoder part convolution
       module_name: RSConvDown # We will be using the RSConvDown Module

       # And provide to each convolution, the associated arguments within a list are selected using the convolution index.
       # For the others, there are just copied for each convolution.
       activation:
           name:  "LeakyReLU"
           negative_slope: 0.2
       ratios: [0.2, 0.25]
       radius: [0.1, 0.2]
       local_nn: [[10, 8, FEAT], [10, 32, 64, 64]]
       down_conv_nn: [[FEAT, 16, 32, 64], [64, 64, 128]]


* First convolution receives ``ratio=0.2``\ , ``radius=0.1``\ , ``local_nn_=[10, 8, 3]``\ , ``down_conv_nn=[3, 16, 32, 64]``
* Second convolution receives ``ratio=0.25``\ , ``radius=0.2``\ , ``local_nn_=[10, 32, 64, 64]``\ , ``down_conv_nn=[64, 64, 128]``
* Both of them will also receive a dictionary ``activation = {name: "LeakyReLU", negative_slope: 0.2}``



Launch an experiment
--------------------
Now that we have our new dataset and model, it is time to launch a training. If you have followed the instructions above you should
be able to simply run the following command and should run smoothly!

.. code-block::

    poetry run python run task=segmentation dataset=s3dis model_type=pointnet2 model_name=pointnet2_charlesmsg

Your terminal should contain:

.. image:: ../imgs/logging.png
   :target: ../imgs/logging.png
   :alt: logging





Train and Test on tasks already implemented
-----------------------

In this section, We will see How we can train and test model on existing datasets.

Registration Task
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In registration task, the goal is to find the right transformation that align correctly pattern.
Here, we will show how we can use deep learning to solve this task. Especially, we will see how we can use `Fully Convolutional Geometric Feature <https://openaccess.thecvf.com/content_ICCV_2019/papers/Choy_Fully_Convolutional_Geometric_Features_ICCV_2019_paper.pdf>`_. FCGF use a Unet architecture to compute feature per point and then we can match these features. Then to find the correct transformation, we can use algorithms such as RANSAC or Fast Global Registration. For this task, we use siamese networks, it means that the dataset provides pairs of point clouds and the networks is applied to both pairs.

To train on 3DMatch, we can type the command:

.. code-block::

   poetry run python train.py task=registration model_type=minkowski model_name=MinkUNet_Fragment dataset=fragment3dmatch_sparse training=sparse_fragment_reg

the config file for models are in the ``conf/models/registration/``. It automatically instantiates models written in ``torch_points3d/models/registration``.
The config file for the datasets are here ``conf/data/registration``. preprocessing and data augmentation are defined here.
So here, it will train a network with the sparse convolution from Minkowski engine, with the architecture specified in the following path on 3DMatch.


We can try an other convolution (for example KPConv):

.. code-block::

   poetry run train.py task=registration model_type=kpconv model_name=KPFCNN dataset=fragment3dmatch_partial training=sparse_fragment_reg``

In the case of KPConv, because it's not the same convolution, the pre-processing is different.
3DMatch is a dataset containing RGBD frames and the poses from 5 different datasets.But our method need to be trained on 3D point cloud. So we need to fuse RGBD frame to create fragments.
Our code will download automatically the RGBD frames with the poses. To build the fragment, we mainly rely on the code from `this repository <https://github.com/andyzeng/tsdf-fusion-python>`_:
In the yaml code, we specify the params to build the fragments for the training and the evaluation and also we provide the parameters for the evaluation.


If you want to test your model you can use the provided scripts.

.. code-block::

   python scripts/test_registration_scripts/evaluate.py task=registration model_type=minkowski model_name=MinkUNet_Fragment dataset=fragment3dmatch_sparse training.checkpoint_dir="your weights " data.sym=True

Where you have to replace "Your weights" by the directory containing the weights.
Finally, if you want to use the networks off the shelf on your own project (using the ``PretrainedRegistry``). you can check the notebooks ``notebooks/demo_registration_3dm.ipynb`` and ``demo_registration_kitti.ipynb`` for 3DMatch and KITTI.
