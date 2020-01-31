# Logging


<h6> The associated logging </h6>

As experiment is empty, it will use hydra naming convention for the experiment
As log_dir is empty, it will use hydra naming convention for the log directory
{path_to_project}/outputs/2019-12-28/12-05-45 (Y-M-D/H-M-S)

The ```name``` is let to the user choose.

<h6>The associated dataset</h6>

```experiment.dataset``` value is used as a key to dynamically choose the associated dataset arguments


<h6> The associated visualization </h6>

The framework currently support both [```wandb```](https://www.wandb.com/) and [```tensorboard```](https://www.tensorflow.org/tensorboard)

```yaml
# parameters for Weights and Biases
wandb:
    project: benchmarking
    log: False

# parameters for TensorBoard Visualization
tensorboard:
    log: True
```