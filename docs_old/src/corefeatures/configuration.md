## Overview

We have chosen [Facebook Hydra library](https://hydra.cc/docs/intro)  as out core tool for managing the configuration of our experiments. It provides a nice and scalable interface to defining models and datasets. We encourage our users to take a look at their documentation and get a basic understanding of its core functionalities.
As per their website 
> "Hydra is a framework for elegantly configuring complex applications"


## Configuration architecture
All configurations leave in the [`./conf`](https://github.com/nicolas-chaulet/deeppointcloud-benchmarks/tree/master/conf) folder and it is organised as follow:
```bash
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
```


## Understanding config.yaml
`config.yaml` is the config file that governs the behaviour of your trainings. It gathers multiple configurations into one, and it is organised as follow:
```yaml
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
```

Hydra is expecting the followings arguments from the command line:

*  task
*  model_type
*  dataset
*  model_name

The provided `task` and `dataset` will be used to load the configuration for the dataset at ```conf/data/{task}/{dataset}.yaml``` while the `model_type` argument will be used to load the model config at ```conf/models/{task}/{model_type}.yaml```.
Finally `model_name` is used to pull the appropriate model from the model configuration file. 

## Training arguments

```yaml
# Those arguments defines the training hyper-parameters
training:
    epochs: 100
    num_workers: 6
    batch_size: 16
    shuffle: True
    cuda: 1
    precompute_multi_scale: False # Compute multiscate features on cpu for faster training / inference
    optim:
        base_lr: 0.001
        # accumulated_gradient: -1 # Accumulate gradient accumulated_gradient * batch_size
        grad_clip: -1
        optimizer:
            class: Adam
            params:
                lr: ${training.optim.base_lr} # The path is cut from training
        lr_scheduler: ${lr_scheduler}
        bn_scheduler:
            bn_policy: "step_decay"
            params:
                bn_momentum: 0.1
                bn_decay: 0.9
                decay_step : 10
                bn_clip : 1e-2
    weight_name: "latest" # Used during resume, select with model to load from [miou, macc, acc..., latest]
    enable_cudnn: True
    checkpoint_dir: ""

# Those arguments within experiment defines which model, dataset and task to be created for benchmarking
# parameters for Weights and Biases
wandb:
    project: default
    log: False
    notes:
    name:
    public: True # It will be display the model within wandb log, else not.

    # parameters for TensorBoard Visualization
tensorboard:
    log: True

```

* ```precompute_multi_scale```: Computes spatial queries such as grid sampling and neighbour search on cpu for faster. Currently this is only supported for KPConv.


## Eval arguments

```yaml
num_workers: 2
batch_size: 16
cuda: 1
weight_name: "latest" # Used during resume, select with model to load from [miou, macc, acc..., latest]
enable_cudnn: True
checkpoint_dir: "" # "{your_path}/outputs/2020-01-28/11-04-13" for example
model_name: KPConvPaper
precompute_multi_scale: True # Compute multiscate features on cpu for faster training / inference
enable_dropout: False
```
