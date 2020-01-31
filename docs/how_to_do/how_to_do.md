
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
```

Hydra is expecting the followings arguments from the command line:

*  task
*  model_type
*  dataset
*  model_name

The provided task and dataset will be used to load within ```conf/data/{task}/{dataset}.yaml``` file within data

The provided task and dataset will be used to load within ```conf/models/{task}/{dataset}.yaml``` file within models

<h4> Training arguments </h4>

```yaml
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

```
* ```weight_name```: Used when ```resume is True```, ```select``` with model to load from ```[metric_name..., latest]```

* ```precompute_multi_scale```: Compute multiscate features on cpu for faster


<h4> Eval arguments </h4>

```yaml
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
```


# Create a new dataset

Let's add [```S3DIS```](http://buildingparser.stanford.edu/dataset.html) dataset to the project.

We are going to go through the successive steps to do so:

*  Choose the associated task related to your dataset.

*  Create a new ```.yaml file``` used for configuring your dataset within ```conf/data/{your_task}/{your_dataset_name}.yaml```

*  Add your own custom configuration needed to parametrize your dataset

*  Create a new file ```src/datasets/{your_task}/{your_dataset}.py```

*  Implement your dataset to inherit from ```BaseDataset```

* Associate a ```metric tracker``` to your dataset.

* Implement your custom ```metric tracker```.

Let's go throught those steps together.

<h4> Choose the associated task for S3DIS </h4>

We are going to focus on semantic segmentation.
Our data are going to be a colored rgb pointcloud associated where each point has been associated to its own class.

The associated task is ```segmentation```.

<h4> Create a new ```.yaml file``` </h4>

Let's create ```conf/data/segmentation/s3dis.yaml``` file.

<h4> Add our own custom configuration </h4>

```yaml
data:
    task: segmentation
    class: s3dis.S3DISDataset
    dataroot: data
    fold: 5
    class_weight_method: "sqrt"
    room_points: 32768
    num_points: 4096
    first_subsampling: 0.04
    density_parameter: 5.0
    kp_extent: 1.0
```

Here, one need note some very important parameters !

* ```task``` needs to be specified. Currently, the arguments provided by the command line are lost and therefore we need the extra information.

* ```class``` needs to be specified. It is structured in the following: {dataset_file}/{dataset_class_name}. In order to create this dataset, we will look into 
```src/datasets/segmentation/s3dis.py``` file and get the ```S3DISDataset``` from it.
The remaining params will be given to the class along the training params.

<h4> Create a new file ```src/datasets/{your_task}/{your_dataset}.py``` </h4>

Now, create a new file ```src/datasets/segmentation/s3dis.py``` with the class ```S3DISDataset``` inside.

<h4>  Implement your dataset to inherit from ```BaseDataset`` </h4>

Before starting, we strongly advice to read the [```Creating Your Own Datasets```](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html) from ```Pytorch Geometric```

```python
class S3DISDataset(BaseDataset):
    def __init__(self, dataset_opt, training_opt):
        super().__init__(dataset_opt, training_opt)
        self._data_path = os.path.join(dataset_opt.dataroot, "S3DIS")

        pre_transform = cT.GridSampling(dataset_opt.first_subsampling, 13)

        transform = T.Compose(
            [T.FixedPoints(dataset_opt.num_points), T.RandomTranslate(0.01), T.RandomRotate(180, axis=2),]
        )

        train_dataset = S3DIS_With_Weights(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=True,
            pre_transform=pre_transform,
            transform=transform,
            class_weight_method=dataset_opt.class_weight_method,
        )
        test_dataset = S3DIS_With_Weights(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=False,
            pre_transform=pre_transform,
            transform=T.FixedPoints(dataset_opt.num_points),
        )

        self._create_dataloaders(train_dataset, test_dataset, val_dataset=None)

    @staticmethod
    def get_tracker(model, task: str, dataset, wandb_opt: bool, tensorboard_opt: bool):
        """Factory method for the tracker

        Arguments:
            task {str} -- task description
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(dataset, wandb_log=wandb_opt.log, use_tensorboard=tensorboard_opt.log)
```

Let's explain the code more in details there.

```python
class S3DISDataset(BaseDataset):
    def __init__(self, dataset_opt, training_opt):
        super().__init__(dataset_opt, training_opt)
        self._data_path = os.path.join(dataset_opt.dataroot, "S3DIS")
```

* We have create a dataset called ```S3DISDataset``` as referenced within our ```s3dis.yaml``` file.

* We can only observe the dataset inherit from ```BaseDataset```. Without it, the new dataset won't be working within the framework !

* ```self._data_path``` will be the place where the data will be saved.

```python
        pre_transform = cT.GridSampling(dataset_opt.first_subsampling, 13)

        transform = T.Compose(
            [T.FixedPoints(dataset_opt.num_points), T.RandomTranslate(0.01), T.RandomRotate(180, axis=2),]
        )

        train_dataset = S3DIS_With_Weights(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=True,
            pre_transform=pre_transform,
            transform=transform,
            class_weight_method=dataset_opt.class_weight_method,
        )
        test_dataset = S3DIS_With_Weights(
            self._data_path,
            test_area=self.dataset_opt.fold,
            train=False,
            pre_transform=pre_transform,
            transform=T.FixedPoints(dataset_opt.num_points),
        )
```

This part creates some transform and train / test dataset.

```python
self._create_dataloaders(train_dataset, test_dataset, val_dataset=None)
```

This line is important. It is going to wrap your datasets directly within the correct dataloader. Don't forget to call this function. Also, we can observe it is possible to provide a ```val_dataset```.

<h4> Associate a ```metric tracker``` to your dataset </h4>

```python
    @staticmethod
    def get_tracker(model, task: str, dataset, wandb_opt: bool, tensorboard_opt: bool):
        """Factory method for the tracker

        Arguments:
            task {str} -- task description
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(dataset, wandb_log=wandb_opt.log, use_tensorboard=tensorboard_opt.log)
```

Finally, one needs to implement the ```@staticmethod get_tracker``` method with ```model, task: str, dataset, wandb_opt: bool, tensorboard_opt: bool``` as parameters.

<h4> Let's have a look at the SegmentationTracker </h4>

```python
class SegmentationTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False):
        """ This is a generic tracker for segmentation tasks.
        It uses a confusion matrix in the back-end to track results.
        Use the tracker to track an epoch.
        You can use the reset function before you start a new epoch

        Arguments:
            dataset  -- dataset to track (used for the number of classes)

        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
            wandb_log {str} --  Log using weight and biases
        """
        super(SegmentationTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self._num_classes = dataset.num_classes
        self.reset(stage)

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._confusion_matrix = ConfusionMatrix(self._num_classes)

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.confusion_matrix

    def track(self, model: BaseModel):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        outputs = self._convert(model.get_output())
        targets = self._convert(model.get_labels())
        assert outputs.shape[0] == len(targets)
        self._confusion_matrix.count_predicted_batch(targets, np.argmax(outputs, 1))

        self._acc = 100 * self._confusion_matrix.get_overall_accuracy()
        self._macc = 100 * self._confusion_matrix.get_mean_class_accuracy()
        self._miou = 100 * self._confusion_matrix.get_average_intersection_union()

    def get_metrics(self, verbose=False) -> Dict[str, float]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_acc".format(self._stage)] = self._acc
        metrics["{}_macc".format(self._stage)] = self._macc
        metrics["{}_miou".format(self._stage)] = self._miou

        return metrics
```

The tracker needs to inherit from the ```BaseTracker``` and implements the following methods:

* ```reset```: The tracker need to be reset when switching to a new stage ```["train", "test", "val"]```

* ```track```: This function is responsible to implement your metrics

* ```get_metrics```: This function is responsible to return a dictionnary with all the tracked metrics for your dataset.


# Convolution Type Introduction

While developing this project, we discovered there are several ways to implement a convolution.

We denoted 4 different formats and they all have their own advantages and drawbacks using two different collate function.
The collate function denotes how a list of examples is converted to a batch

* [```torch.utils.data.DataLoader```](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html)
    - ```"DENSE"```

* [```torch_geometric.data.DataLoader```](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset)
    - ```"PARTIAL_DENSE"```
    - ```"MESSAGE_PASSING"```
    - ```"SPARSE"```

<h4> DENSE ConvType Format with torch.utils.data.DataLoader</h4>

The N tensor of shape (num_points, feat_dim) will be concatenated on a new dimension
[(num_points, feat_dim), ..., (num_points, feat_dim)] -> (N, num_points, feat_dim)

This format forces each tensor to have exactly the same shape

Advantages

* The format is dense and therefore aggregation operation are fast

Drawbacks

* When ```neighbour finder``` is applied. Each point is going to be forced to have         ```max_num_neighbors: default = 64```. Therefore, they will lot of shadow points to complete the batch_size, creating extra useless memory consumption

<h2> torch_geometric.data.DataLoader TYPE </h2>

![Screenshot](/imgs/pyg_batch.PNG)

The collate function can be found there [```Batch.from_data_list```](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/batch.html#Batch)

Given ```N tensors``` with their own ```num_points_{i}```, the ```collate function``` does:
```[(num_points_1, feat_dim), ..., (num_points_n, feat_dim)] -> (num_points_1 + ... + num_points_n, feat_dim)```

It also creates an associated ```batch tensor``` of size ```(num_points_1 + ... + num_points_n)``` with indices of the corresponding batch.

```Example```:

* A with shape (2, 2)
* B with shape (3, 2)

```C = Batch.from_data_list([A, B])```

C is a tensor of shape ```(5, 2)``` and its associated batch will contain ```[0, 0, 1, 1, 1]```


<h4> PARTIAL_DENSE ConvType Format </h4>

This format is used by KPConv original implementation.


Same as dense format, it forces each point to have the same number of neighbors.
It is why we called it partially dense.

<h4> MESSAGE_PASSING ConvType Format </h4>

This ConvType is Pytorch Geometric base format.
Using [```Message Passing```](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/message_passing.html#MessagePassing) API class, it deploys the graph created by ```neighbour finder``` using internally the ```torch.index_select``` operator.

Therefore, the ```[PointNet++]``` internal convolution looks like that.

```python
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
```

<h4> SPARSE ConvType Format </h4>


| Dense Tensor                                                                    | Sparse Tensor                                                                     |
|:-------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|
| <img src="https://stanfordvl.github.io/MinkowskiEngine/_images/conv_dense.gif"> | <img src="https://stanfordvl.github.io/MinkowskiEngine/_images/conv_sparse.gif" > |

The sparse conv type is used by project like [```SparseConv```](https://github.com/facebookresearch/SparseConvNet) or [```Minkowski Engine```](https://github.com/StanfordVL/MinkowskiEngine)

Therefore, the points have to be converted into indices living within a grid.

# Create a new model

Let's add [```PointNet++```](https://github.com/charlesq34/pointnet2) model implemented within the "DENSE" format type to the project.

We are going to go through the successive steps to do so:

*  Choose the associated task related to your new model.
    - ```PointNet++``` modules could be used for different task
    - ```PointNet++``` models are task related.

*  Create a new ```.yaml file``` used for configuring your dataset within ```conf/models/{your_task}/{your_model_name}.yaml```

*  Add your own custom configuration needed to parametrize your model

*  Create a new file ```src/datasets/{your_task}/{your_dataset}.py```
    - ```PointNet++``` configuration could be pretty different from other models !

*  Implement your model to inherit from ```BaseModel```

<h4> MESSAGE_PASSING ConvType Format </h4>

