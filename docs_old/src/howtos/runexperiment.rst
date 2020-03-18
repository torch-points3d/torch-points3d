
launch an experiment with your new models and datasets
------------------------------------------------------

.. code-block::

   poetry run python run task=segmentation dataset=s3dis model_type=pointnet2 model_name=pointnet2_charlesmsg

Your terminal shoud contain:

.. code-block::


   # Understand hydra configuration

   We recommend people willing to use the framework to get familiries with [```Facebook Hydra library```](https://hydra.cc/docs/intro).

   Reading quickly through Hydra documentation should give one the basic understanding of its core functionalites.

   To make it short, it is [```argparse```](https://docs.python.org/2/library/argparse.html) built on top of yaml file, allowing ```arguments to be defined in a tree structure```.


   <h2>Configuration architecture</h2>

   * config.yaml
       * hydra/ # configuration related to hydra
       * models/ # configuration related to the models
       * data/ # configuration related to the datasets
   * training.yaml
   * eval.yaml


   <h4>Understanding config.yaml</h4>

   ```yaml
   defaults:
     - task: ??? # Task performed (segmentation, classification etc...)
       optional: True
     - model_type: ??? # Type of model to use, e.g. pointnet2, rsconv etc...
       optional: True
     - dataset: ???
       optional: True

     - training
     - eval

     - models: ${defaults.0.task}/${defaults.1.model_type}
     - data: ${defaults.0.task}/${defaults.2.dataset}
     - sota # Contains current SOTA results on different datasets (extracted from papers !).
     - hydra/job_logging: custom

   model_name: ??? # Name of the specific model to load

   # Those arguments within experiment defines which model, dataset and task to be created for benchmarking
   # parameters for Weights and Biases
   wandb:
     project: shapenet-pn2
     log: False

   # parameters for TensorBoard Visualization
   tensorboard:
     log: True

Hydra is expecting the followings arguments from the command line:


* task
* model_type
* dataset
* model_name

The provided task and dataset will be used to load within ``conf/data/{task}/{dataset}.yaml`` file within data

The provided task and dataset will be used to load within ``conf/models/{task}/{dataset}.yaml`` file within models


.. raw:: html

   <h4> Training arguments </h4>


.. code-block:: yaml

   training:
       shuffle: True
       num_workers: 2
       batch_size: 16
       cuda: 1
       precompute_multi_scale: False # Compute multiscate features on cpu for faster training / inference
       epochs: 100
       optimizer: Adam
       learning_rate:
           scheduler_type: "step_decay"
           base_lr: 0.001
           lr_decay: 0.5
           decay_step: 200000
           lr_clip: 1e-5
       weight_name: "latest" # Used during resume, select with model to load from [miou, macc, acc..., latest]
       enable_cudnn: True
       checkpoint_dir: ""
       resume: True


* 
  ``weight_name``\ : Used when ``resume is True``\ , ``select`` with model to load from ``[metric_name..., latest]``

* 
  ``precompute_multi_scale``\ : Compute multiscate features on cpu for faster


.. raw:: html

   <h4> Eval arguments </h4>


.. code-block:: yaml

   eval:
       shuffle: True
       num_workers: 2
       batch_size: 1
       cuda: 1
       weight_name: "latest" # Used during resume, select with model to load from [miou, macc, acc..., latest]
       enable_cudnn: True
       checkpoint_dir: "" # "{your_path}/outputs/2020-01-28/11-04-13" for example
       precompute_multi_scale: False # Compute multiscate features on cpu for faster training / inference
       enable_dropout: True # Could be used from MC dropout sampling
