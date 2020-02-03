
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

# Create a new model

Let's add [```PointNet++```](https://github.com/charlesq34/pointnet2) model implemented within the "DENSE" format type to the project.

We are going to go through the successive steps to do so:

*  Choose the associated task related to your new model.
    - ```PointNet++``` modules could be used for different task
    - ```PointNet++``` models are task related.

*  Create a new ```.yaml file``` used for configuring your model within ```conf/models/{your_task}/{your_model_name}.yaml```

*  Add your own custom configuration needed to parametrize your model

*  Create a new file ```src/models/{your_task}/{your_model}.py```
    - ```PointNet++``` configuration could be pretty different from other models !

*  Implement your model to inherit from ```BaseModel```

*  Create a new folder within modules ```src/modules/{model_type}/{conv_type}.py```

*  Implement ```PointNet++``` associated modules with this file.


<h4> Choose the associated task related to your new model </h4>

We are going to implement ```PointNet++``` for ```segmentation``` task on ```S3DIS```.

<h4> Create a new .yaml file used for configuring your model </h4>

We create a new file ```conf/models/segmentation/pointnet2.yaml```.
This file will contain all the ```different versions``` of pointnet++.

```python
models:
    pointnet2_charlesmsg:
        architecture: pointnet2.PointNet2_D
        conv_type: "DENSE"
        use_category: True
        down_conv:
            module_name: PointNetMSGDown
            npoint: [512, 128]
            radii: [[0.1, 0.2, 0.4], [0.4, 0.8]]
            nsamples: [[32, 64, 128], [64, 128]]
            down_conv_nn:
                [
                    [
                        [FEAT, 32, 32, 64],
                        [FEAT, 64, 64, 128],
                        [FEAT, 64, 96, 128],
                    ],
                    [
                        [64 + 128 + 128, 128, 128, 256],
                        [64 + 128 + 128, 128, 196, 256],
                    ],
                ]
        innermost:
            module_name: GlobalDenseBaseModule
            nn: [256 * 2 + 3, 256, 512, 1024]
        up_conv:
            module_name: DenseFPModule
            up_conv_nn:
                [
                    [1024 + 256*2, 256, 256],
                    [256 + 128 * 2 + 64, 256, 128],
                    [128 + FEAT, 128, 128],
                ]
            skip: True
        mlp_cls:
            nn: [128, 128]
            dropout: 0.5
```

Here is ```PointNet++``` Multi-Scale original version by [```Charles R. Qi```](https://github.com/charlesq34).

Let's dig in the definition.

<h6> Requiered arguments </h6>

* ```pointnet2_charlesmsg``` is model_name and should be provided from the commande line in order to load this file configuration.

* ```architecture: pointnet2.PointNet2_D```. It indicates where to find the Model Logic.
The framework backend will look for the file ```/src/models/segmentation/pointnet2.py``` and the ```PointNet2_D``` class.

* ```conv_type: "DENSE"```

<h6> "Optional" arguments </h6>

When I say optional, I mean those parameters could be defined differently for your own model.
We don't want to force any particular configuration format.
```However, the simplest is always better !```

This particular format is used by our  [```Unet architecture```](https://arxiv.org/abs/1505.04597) builder base class [```src/models/base_architectures/unet.py```](https://github.com/nicolas-chaulet/deeppointcloud-benchmarks/blob/master/src/models/base_architectures/unet.py) with ```UnetBasedModel``` and ```UnwrappedUnetBasedModel```.

Those particular class is looking for those keys within the configuration

* ```down_conv```
* ```innermost```
* ```up_conv```

Those elements need to contain a ```module_name``` which will be used to create the associated Module.

Those BaseUnets class will do the followings:

* If provided a list, it will use the index to access the value
* If provided something else, it will broadcast the arguments to all convolutions.

Here is an example with the ```RSConv``` implementation in ```MESSAGE_TYPE ConvType```.

```python
class RSConv(BaseConvolutionDown):
    def __init__(self, ratio=None, radius=None, local_nn=None, down_conv_nn=None, nb_feature=None, *args, **kwargs):
        super(RSConv, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)
        
        self._conv = Convolution(local_nn=local_nn, global_nn=down_conv_nn)

    def conv(self, x, pos, edge_index, batch):
        return self._conv(x, pos, edge_index)

```
We can see this convolution needs the followings arguments
```python
ratio=None, radius=None, local_nn=None, down_conv_nn=None, nb_feature=None
```
Here is an extract from the model architecture config:

```yaml
down_conv: # For the encoder part convolution
    module_name: RSConv # We will be using the RSConv Module
    
    # And provide to each convolution, the associated arguments within a list are selected using the convolution index.
    # For the others, there are just copied for each convolution.
    activation:
        name:  "LeakyReLU"
        negative_slope: 0.2
    ratios: [0.2, 0.25] 
    radius: [0.1, 0.2]
    local_nn: [[10, 8, FEAT], [10, 32, 64, 64]]
    down_conv_nn: [[FEAT, 16, 32, 64], [64, 64, 128]]
```

* First convolution receives ```ratio=0.2```, ```radius=0.1```, ```local_nn_=[10, 8, 3]```, ```down_conv_nn=[3, 16, 32, 64]```
* Second convolution receives ```ratio=0.25```, ```radius=0.2```, ```local_nn_=[10, 32, 64, 64]```, ```down_conv_nn=[64, 64, 128]```
* Both of them will also receive a dictionary ```activation = {name: "LeakyReLU", negative_slope: 0.2}``` 

<h4> Create a new file [src/models/{your_task}/{your_model}.py] </h4>

Let's create a new file ```/src/models/segmentation/pointnet2.py``` with its associated class 
```PointNet2_D```

```python
import torch

import torch.nn.functional as F
from torch_geometric.data import Data
import etw_pytorch_utils as pt_utils
import logging

from src.modules.pointnet2 import * # This part is extremely important. Always important the associated modules within your this file
from src.core.base_conv.dense import DenseFPModule
from src.models.base_architectures import UnetBasedModel

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

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
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

        if torch.isnan(self.loss_seg):
            import pdb

            pdb.set_trace()
        self.loss_seg.backward()  # calculate gradients of network G w.r.t. loss_G
```

*  IT IS IMPORTANT TO ALWAYS IMPORT ALL THE NEEDED MODULES TO CONSTRUCT THE MODEL.

* IT IS IMPORTANT TO ALWAYS INHERIT YOUR MODEL FROM ```BASEMODEL```.

Here is the ```BaseModel``` you can find within ```/src/models/base_model.py```

```python
class BaseModel(torch.nn.Module):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        super(BaseModel, self).__init__()
        self.opt = opt
        self.loss_names = []
        self.output = None
        self._optimizer: Optional[Optimizer] = None
        self._lr_scheduler: Optimizer[_LRScheduler] = None
        self._sampling_and_search_dict: Dict = {}
        self._precompute_multi_scale = opt.precompute_multi_scale if "precompute_multi_scale" in opt else False
        self._iterations = 0
        self._lr_params = None

    @property
    def lr_params(self):
        try:
            params = copy.deepcopy(self._lr_params)
            params.lr_base = self.learning_rate
            return params
        except:
            return None

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """

    def get_labels(self):
        """ returns a trensor of size [N_points] where each value is the label of a point
        """
        return getattr(self, "labels", None)

    def get_batch_idx(self):
        """ returns a trensor of size [N_points] where each value is the batch index of a point
        """
        return getattr(self, "batch_idx", None)

    def get_output(self):
        """ returns a trensor of size [N_points,...] where each value is the output
        of the network for a point (output of the last layer in general)
        """
        return self.output

    @abstractmethod
    def forward(self) -> Any:
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

    def optimize_parameters(self, batch_size):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self._iterations += batch_size
        self.forward()  # first call forward to calculate intermediate results
        self._optimizer.zero_grad()  # clear existing gradients
        self.backward()  # calculate gradients
        self._optimizer.step()  # update parameters
        if self._lr_scheduler is not None:
            self._lr_scheduler.step(self._iterations)

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                if hasattr(self, name):
                    try:
                        errors_ret[name] = float(getattr(self, name))
                    except:
                        errors_ret[name] = None
        return errors_ret

    def set_optimizer(self, optimizer_cls: Optimizer, lr_params):
        self._optimizer = optimizer_cls(self.parameters(), lr=lr_params.base_lr)
        self._lr_scheduler = get_scheduler(lr_params, self._optimizer)
        self._lr_params = lr_params
        log.info(self._optimizer)

    def get_named_internal_losses(self):
        """
            Modules which have internal losses return a dict of the form
            {<loss_name>: <loss>}
            This method merges the dicts of all child modules with internal loss
            and returns this merged dict
        """

        losses_global = []

        def search_from_key(modules, losses_global):
            for _, module in modules.items():
                if isinstance(module, BaseInternalLossModule):
                    losses_global.append(module.get_internal_losses())
                search_from_key(module._modules, losses_global)

        search_from_key(self._modules, losses_global)

        return dict(ChainMap(*losses_global))

    def get_internal_loss(self):
        """
            Returns the average internal loss of all child modules with
            internal losses
        """

        losses = tuple(self.get_named_internal_losses().values())
        if len(losses) > 0:
            return torch.mean(torch.stack(losses))
        else:
            return 0.0

    def get_sampling_and_search_strategies(self):
        return self._sampling_and_search_dict

    def enable_dropout_in_eval(self):
        def search_from_key(modules):
            for _, m in modules.items():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()
                search_from_key(m._modules)

        search_from_key(self._modules)
```

<h4> Implement PointNet++ Dense modules src/modules/{model_type}/{conv_type}.py </h4>

Let's create ```src/modules/pointnet2/``` directory and ```dense.py``` file within.

! NOTE: Remember to create an ```__init__.py file``` and add ```from .dense import *```.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_points as tp
import etw_pytorch_utils as pt_utils

from src.core.base_conv.dense import *
from src.core.neighbourfinder import DenseRadiusNeighbourFinder
from src.core.sampling import DenseFPSSampler
from src.utils.model_building_utils.activation_resolver import get_activation


class PointNetMSGDown(BaseDenseConvolutionDown):
    def __init__(
        self,
        npoint=None,
        radii=None,
        nsample=None,
        down_conv_nn=None,
        bn=True,
        activation="LeakyReLU",
        use_xyz=True,
        **kwargs
    ):
        assert len(radii) == len(nsample) == len(down_conv_nn)
        super(PointNetMSGDown, self).__init__(
            DenseFPSSampler(num_to_sample=npoint), DenseRadiusNeighbourFinder(radii, nsample), **kwargs
        )
        self.use_xyz = use_xyz
        self.npoint = npoint
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            mlp_spec = down_conv_nn[i]
            if self.use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_utils.SharedMLP(down_conv_nn[i], bn=bn, activation=get_activation(activation)))

    def _prepare_features(self, x, pos, new_pos, idx):
        new_pos_trans = pos.transpose(1, 2).contiguous()
        grouped_pos = tp.grouping_operation(new_pos_trans, idx)  # (B, 3, npoint, nsample)
        grouped_pos -= new_pos.transpose(1, 2).unsqueeze(-1)

        if x is not None:
            grouped_features = tp.grouping_operation(x, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_pos, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_pos

        return new_features

    def conv(self, x, pos, new_pos, radius_idx, scale_idx):
        """ Implements a Dense convolution where radius_idx represents
        the indexes of the points in x and pos to be agragated into the new feature
        for each point in new_pos

        Arguments:
            x -- Previous features [B, N, C]
            pos -- Previous positions [B, N, 3]
            new_pos  -- Sampled positions [B, npoints, 3]
            radius_idx -- Indexes to group [B, npoints, nsample]
            scale_idx -- Scale index in multiscale convolutional layers
        Returns:
            new_x -- Features after passing trhough the MLP [B, mlp[-1], npoints]
        """
        assert scale_idx < len(self.mlps)
        new_features = self._prepare_features(x, pos, new_pos, radius_idx)
        new_features = self.mlps[scale_idx](new_features)  # (B, mlp[-1], npoint, nsample)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
        return new_features
```

Here is the MultiScale Dense implementation of PointNet++.
Let's dig in.

```python
class PointNetMSGDown(BaseDenseConvolutionDown):
    def __init__(
        ...
    ):
        super(PointNetMSGDown, self).__init__(
            DenseFPSSampler(num_to_sample=npoint), DenseRadiusNeighbourFinder(radii, nsample), **kwargs
        )
```

* The ```PointNetMSGDown``` inherit from ```BaseDenseConvolutionDown```:
    - ```BaseDenseConvolutionDown``` takes care of all the sampling and search logic for you.
    - Therefore, a ```sampler``` and a ```neighbour finder``` has to be provided.
    - Here, we provide ```DenseFPSSampler``` and ```DenseRadiusNeighbourFinder```

* The ```PointNetMSGDown``` class just needs to implement the ```conv method```.

```python
def conv(self, x, pos, new_pos, radius_idx, scale_idx):
    """ Implements a Dense convolution where radius_idx represents
    the indexes of the points in x and pos to be agragated into the new feature
    for each point in new_pos

    Arguments:
        x -- Previous features [B, N, C]
        pos -- Previous positions [B, N, 3]
        new_pos  -- Sampled positions [B, npoints, 3]
        radius_idx -- Indexes to group [B, npoints, nsample]
        scale_idx -- Scale index in multiscale convolutional layers
    Returns:
        new_x -- Features after passing trhough the MLP [B, mlp[-1], npoints]
    """

    # Do something here ...

    return new_features
```

# Running new model on new dataset


```
poetry run python run task=segmentation dataset=s3dis model_type=pointnet2 model_name=pointnet2_charlesmsg
```

Your terminal shoud contain:
```

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

# Create a new model

Let's add [```PointNet++```](https://github.com/charlesq34/pointnet2) model implemented within the "DENSE" format type to the project.

We are going to go through the successive steps to do so:

*  Choose the associated task related to your new model.
    - ```PointNet++``` modules could be used for different task
    - ```PointNet++``` models are task related.

*  Create a new ```.yaml file``` used for configuring your model within ```conf/models/{your_task}/{your_model_name}.yaml```

*  Add your own custom configuration needed to parametrize your model

*  Create a new file ```src/models/{your_task}/{your_model}.py```
    - ```PointNet++``` configuration could be pretty different from other models !

*  Implement your model to inherit from ```BaseModel```

*  Create a new folder within modules ```src/modules/{model_type}/{conv_type}.py```

*  Implement ```PointNet++``` associated modules with this file.


<h4> Choose the associated task related to your new model </h4>

We are going to implement ```PointNet++``` for ```segmentation``` task on ```S3DIS```.

<h4> Create a new .yaml file used for configuring your model </h4>

We create a new file ```conf/models/segmentation/pointnet2.yaml```.
This file will contain all the ```different versions``` of pointnet++.

```python
models:
    pointnet2_charlesmsg:
        architecture: pointnet2.PointNet2_D
        conv_type: "DENSE"
        use_category: True
        down_conv:
            module_name: PointNetMSGDown
            npoint: [512, 128]
            radii: [[0.1, 0.2, 0.4], [0.4, 0.8]]
            nsamples: [[32, 64, 128], [64, 128]]
            down_conv_nn:
                [
                    [
                        [FEAT, 32, 32, 64],
                        [FEAT, 64, 64, 128],
                        [FEAT, 64, 96, 128],
                    ],
                    [
                        [64 + 128 + 128, 128, 128, 256],
                        [64 + 128 + 128, 128, 196, 256],
                    ],
                ]
        innermost:
            module_name: GlobalDenseBaseModule
            nn: [256 * 2 + 3, 256, 512, 1024]
        up_conv:
            module_name: DenseFPModule
            up_conv_nn:
                [
                    [1024 + 256*2, 256, 256],
                    [256 + 128 * 2 + 64, 256, 128],
                    [128 + FEAT, 128, 128],
                ]
            skip: True
        mlp_cls:
            nn: [128, 128]
            dropout: 0.5
```

Here is ```PointNet++``` Multi-Scale original version by [```Charles R. Qi```](https://github.com/charlesq34).

Let's dig in the definition.

<h6> Requiered arguments </h6>

* ```pointnet2_charlesmsg``` is model_name and should be provided from the commande line in order to load this file configuration.

* ```architecture: pointnet2.PointNet2_D```. It indicates where to find the Model Logic.
The framework backend will look for the file ```/src/models/segmentation/pointnet2.py``` and the ```PointNet2_D``` class.

* ```conv_type: "DENSE"```

<h6> "Optional" arguments </h6>

When I say optional, I mean those parameters could be defined differently for your own model.
We don't want to force any particular configuration format.
```However, the simplest is always better !```

This particular format is used by our  [```Unet architecture```](https://arxiv.org/abs/1505.04597) builder base class [```src/models/base_architectures/unet.py```](https://github.com/nicolas-chaulet/deeppointcloud-benchmarks/blob/master/src/models/base_architectures/unet.py) with ```UnetBasedModel``` and ```UnwrappedUnetBasedModel```.

Those particular class is looking for those keys within the configuration

* ```down_conv```
* ```innermost```
* ```up_conv```

Those elements need to contain a ```module_name``` which will be used to create the associated Module.

Those BaseUnets class will do the followings:

* If provided a list, it will use the index to access the value
* If provided something else, it will broadcast the arguments to all convolutions.

Here is an example with the ```RSConv``` implementation in ```MESSAGE_TYPE ConvType```.

```python
class RSConv(BaseConvolutionDown):
    def __init__(self, ratio=None, radius=None, local_nn=None, down_conv_nn=None, nb_feature=None, *args, **kwargs):
        super(RSConv, self).__init__(FPSSampler(ratio), RadiusNeighbourFinder(radius), *args, **kwargs)
        
        self._conv = Convolution(local_nn=local_nn, global_nn=down_conv_nn)

    def conv(self, x, pos, edge_index, batch):
        return self._conv(x, pos, edge_index)

```
We can see this convolution needs the followings arguments
```python
ratio=None, radius=None, local_nn=None, down_conv_nn=None, nb_feature=None
```
Here is an extract from the model architecture config:

```yaml
down_conv: # For the encoder part convolution
    module_name: RSConv # We will be using the RSConv Module
    
    # And provide to each convolution, the associated arguments within a list are selected using the convolution index.
    # For the others, there are just copied for each convolution.
    activation:
        name:  "LeakyReLU"
        negative_slope: 0.2
    ratios: [0.2, 0.25] 
    radius: [0.1, 0.2]
    local_nn: [[10, 8, FEAT], [10, 32, 64, 64]]
    down_conv_nn: [[FEAT, 16, 32, 64], [64, 64, 128]]
```

* First convolution receives ```ratio=0.2```, ```radius=0.1```, ```local_nn_=[10, 8, 3]```, ```down_conv_nn=[3, 16, 32, 64]```
* Second convolution receives ```ratio=0.25```, ```radius=0.2```, ```local_nn_=[10, 32, 64, 64]```, ```down_conv_nn=[64, 64, 128]```
* Both of them will also receive a dictionary ```activation = {name: "LeakyReLU", negative_slope: 0.2}``` 

<h4> Create a new file [src/models/{your_task}/{your_model}.py] </h4>

Let's create a new file ```/src/models/segmentation/pointnet2.py``` with its associated class 
```PointNet2_D```

```python
import torch

import torch.nn.functional as F
from torch_geometric.data import Data
import etw_pytorch_utils as pt_utils
import logging

from src.modules.pointnet2 import * # This part is extremely important. Always important the associated modules within your this file
from src.core.base_conv.dense import DenseFPModule
from src.models.base_architectures import UnetBasedModel

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

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
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

        if torch.isnan(self.loss_seg):
            import pdb

            pdb.set_trace()
        self.loss_seg.backward()  # calculate gradients of network G w.r.t. loss_G
```

*  IT IS IMPORTANT TO ALWAYS IMPORT ALL THE NEEDED MODULES TO CONSTRUCT THE MODEL.

* IT IS IMPORTANT TO ALWAYS INHERIT YOUR MODEL FROM ```BASEMODEL```.

Here is the ```BaseModel``` you can find within ```/src/models/base_model.py```

```python
class BaseModel(torch.nn.Module):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        super(BaseModel, self).__init__()
        self.opt = opt
        self.loss_names = []
        self.output = None
        self._optimizer: Optional[Optimizer] = None
        self._lr_scheduler: Optimizer[_LRScheduler] = None
        self._sampling_and_search_dict: Dict = {}
        self._precompute_multi_scale = opt.precompute_multi_scale if "precompute_multi_scale" in opt else False
        self._iterations = 0
        self._lr_params = None

    @property
    def lr_params(self):
        try:
            params = copy.deepcopy(self._lr_params)
            params.lr_base = self.learning_rate
            return params
        except:
            return None

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """

    def get_labels(self):
        """ returns a trensor of size [N_points] where each value is the label of a point
        """
        return getattr(self, "labels", None)

    def get_batch_idx(self):
        """ returns a trensor of size [N_points] where each value is the batch index of a point
        """
        return getattr(self, "batch_idx", None)

    def get_output(self):
        """ returns a trensor of size [N_points,...] where each value is the output
        of the network for a point (output of the last layer in general)
        """
        return self.output

    @abstractmethod
    def forward(self) -> Any:
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

    def optimize_parameters(self, batch_size):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self._iterations += batch_size
        self.forward()  # first call forward to calculate intermediate results
        self._optimizer.zero_grad()  # clear existing gradients
        self.backward()  # calculate gradients
        self._optimizer.step()  # update parameters
        if self._lr_scheduler is not None:
            self._lr_scheduler.step(self._iterations)

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                if hasattr(self, name):
                    try:
                        errors_ret[name] = float(getattr(self, name))
                    except:
                        errors_ret[name] = None
        return errors_ret

    def set_optimizer(self, optimizer_cls: Optimizer, lr_params):
        self._optimizer = optimizer_cls(self.parameters(), lr=lr_params.base_lr)
        self._lr_scheduler = get_scheduler(lr_params, self._optimizer)
        self._lr_params = lr_params
        log.info(self._optimizer)

    def get_named_internal_losses(self):
        """
            Modules which have internal losses return a dict of the form
            {<loss_name>: <loss>}
            This method merges the dicts of all child modules with internal loss
            and returns this merged dict
        """

        losses_global = []

        def search_from_key(modules, losses_global):
            for _, module in modules.items():
                if isinstance(module, BaseInternalLossModule):
                    losses_global.append(module.get_internal_losses())
                search_from_key(module._modules, losses_global)

        search_from_key(self._modules, losses_global)

        return dict(ChainMap(*losses_global))

    def get_internal_loss(self):
        """
            Returns the average internal loss of all child modules with
            internal losses
        """

        losses = tuple(self.get_named_internal_losses().values())
        if len(losses) > 0:
            return torch.mean(torch.stack(losses))
        else:
            return 0.0

    def get_sampling_and_search_strategies(self):
        return self._sampling_and_search_dict

    def enable_dropout_in_eval(self):
        def search_from_key(modules):
            for _, m in modules.items():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()
                search_from_key(m._modules)

        search_from_key(self._modules)
```

<h4> Implement PointNet++ Dense modules src/modules/{model_type}/{conv_type}.py </h4>

Let's create ```src/modules/pointnet2/``` directory and ```dense.py``` file within.

! NOTE: Remember to create an ```__init__.py file``` and add ```from .dense import *```.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_points as tp
import etw_pytorch_utils as pt_utils

from src.core.base_conv.dense import *
from src.core.neighbourfinder import DenseRadiusNeighbourFinder
from src.core.sampling import DenseFPSSampler
from src.utils.model_building_utils.activation_resolver import get_activation


class PointNetMSGDown(BaseDenseConvolutionDown):
    def __init__(
        self,
        npoint=None,
        radii=None,
        nsample=None,
        down_conv_nn=None,
        bn=True,
        activation="LeakyReLU",
        use_xyz=True,
        **kwargs
    ):
        assert len(radii) == len(nsample) == len(down_conv_nn)
        super(PointNetMSGDown, self).__init__(
            DenseFPSSampler(num_to_sample=npoint), DenseRadiusNeighbourFinder(radii, nsample), **kwargs
        )
        self.use_xyz = use_xyz
        self.npoint = npoint
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            mlp_spec = down_conv_nn[i]
            if self.use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_utils.SharedMLP(down_conv_nn[i], bn=bn, activation=get_activation(activation)))

    def _prepare_features(self, x, pos, new_pos, idx):
        new_pos_trans = pos.transpose(1, 2).contiguous()
        grouped_pos = tp.grouping_operation(new_pos_trans, idx)  # (B, 3, npoint, nsample)
        grouped_pos -= new_pos.transpose(1, 2).unsqueeze(-1)

        if x is not None:
            grouped_features = tp.grouping_operation(x, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_pos, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_pos

        return new_features

    def conv(self, x, pos, new_pos, radius_idx, scale_idx):
        """ Implements a Dense convolution where radius_idx represents
        the indexes of the points in x and pos to be agragated into the new feature
        for each point in new_pos

        Arguments:
            x -- Previous features [B, N, C]
            pos -- Previous positions [B, N, 3]
            new_pos  -- Sampled positions [B, npoints, 3]
            radius_idx -- Indexes to group [B, npoints, nsample]
            scale_idx -- Scale index in multiscale convolutional layers
        Returns:
            new_x -- Features after passing trhough the MLP [B, mlp[-1], npoints]
        """
        assert scale_idx < len(self.mlps)
        new_features = self._prepare_features(x, pos, new_pos, radius_idx)
        new_features = self.mlps[scale_idx](new_features)  # (B, mlp[-1], npoint, nsample)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
        return new_features
```

Here is the MultiScale Dense implementation of PointNet++.
Let's dig in.

```python
class PointNetMSGDown(BaseDenseConvolutionDown):
    def __init__(
        ...
    ):
        super(PointNetMSGDown, self).__init__(
            DenseFPSSampler(num_to_sample=npoint), DenseRadiusNeighbourFinder(radii, nsample), **kwargs
        )
```

* The ```PointNetMSGDown``` inherit from ```BaseDenseConvolutionDown```:
    - ```BaseDenseConvolutionDown``` takes care of all the sampling and search logic for you.
    - Therefore, a ```sampler``` and a ```neighbour finder``` has to be provided.
    - Here, we provide ```DenseFPSSampler``` and ```DenseRadiusNeighbourFinder```

* The ```PointNetMSGDown``` class just needs to implement the ```conv method```.

```python
def conv(self, x, pos, new_pos, radius_idx, scale_idx):
    """ Implements a Dense convolution where radius_idx represents
    the indexes of the points in x and pos to be agragated into the new feature
    for each point in new_pos

    Arguments:
        x -- Previous features [B, N, C]
        pos -- Previous positions [B, N, 3]
        new_pos  -- Sampled positions [B, npoints, 3]
        radius_idx -- Indexes to group [B, npoints, nsample]
        scale_idx -- Scale index in multiscale convolutional layers
    Returns:
        new_x -- Features after passing trhough the MLP [B, mlp[-1], npoints]
    """

    # Do something here ...

    return new_features
```

# Running new model on new dataset


```
poetry run python run task=segmentation dataset=s3dis model_type=pointnet2 model_name=pointnet2_charlesmsg
```

Your terminal shoud contain:
```
(deeppointcloud-benchmark-VS2SMlit-py3.6) thoma@thomas-devmachine:~/deeppointcloud-benchmarks$ poetry run python train.py task=segmentation dataset=s3dis1x1 model_type=pointnet2 model_name=pointnet2_charlesssg
DEVICE : cuda
CLASS WEIGHT : {'ceiling': 0.0249, 'floor': 0.026, 'wall': 0.0301, 'column': 0.0805, 'beam': 0.1004, 'window': 0.1216, 'door': 0.0584, 'table': 0.0679, 'chair': 0.0542, 'bookcase': 0.179, 'sofa': 0.069, 'board': 0.1509, 'clutter': 0.0371}
PointNet2_D(
  (model): UnetSkipConnectionBlock(
    (down): PointNetMSGDown(
      (mlps): ModuleList(
        (0): SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(9, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): LeakyReLU(negative_slope=0.01)
          )
          (layer1): Conv2d(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): LeakyReLU(negative_slope=0.01)
          )
          (layer2): Conv2d(
            (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): LeakyReLU(negative_slope=0.01)
          )
        )
      )
    )
    (submodule): UnetSkipConnectionBlock(
      (down): PointNetMSGDown(
        (mlps): ModuleList(
          (0): SharedMLP(
            (layer0): Conv2d(
              (conv): Conv2d(131, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (activation): LeakyReLU(negative_slope=0.01)
            )
            (layer1): Conv2d(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (activation): LeakyReLU(negative_slope=0.01)
            )
            (layer2): Conv2d(
              (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (normlayer): BatchNorm2d(
                (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (activation): LeakyReLU(negative_slope=0.01)
            )
          )
        )
      )
      (submodule): UnetSkipConnectionBlock(
        (inner): GlobalDenseBaseModule: 725248 (aggr=max, SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(259, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): LeakyReLU(negative_slope=0.01)
          )
          (layer1): Conv2d(
            (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): LeakyReLU(negative_slope=0.01)
          )
          (layer2): Conv2d(
            (conv): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): LeakyReLU(negative_slope=0.01)
          )
        ))
        (up): DenseFPModule: 394240 (SharedMLP(
          (layer0): Conv2d(
            (conv): Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): LeakyReLU(negative_slope=0.01)
          )
          (layer1): Conv2d(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (normlayer): BatchNorm2d(
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (activation): LeakyReLU(negative_slope=0.01)
          )
        ))
      )
      (up): DenseFPModule: 131840 (SharedMLP(
        (layer0): Conv2d(
          (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normlayer): BatchNorm2d(
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): LeakyReLU(negative_slope=0.01)
        )
        (layer1): Conv2d(
          (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normlayer): BatchNorm2d(
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): LeakyReLU(negative_slope=0.01)
        )
      ))
    )
    (up): DenseFPModule: 50688 (SharedMLP(
      (layer0): Conv2d(
        (conv): Conv2d(134, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normlayer): BatchNorm2d(
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): LeakyReLU(negative_slope=0.01)
      )
      (layer1): Conv2d(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normlayer): BatchNorm2d(
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): LeakyReLU(negative_slope=0.01)
      )
      (layer2): Conv2d(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normlayer): BatchNorm2d(
          (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (activation): LeakyReLU(negative_slope=0.01)
      )
    ))
  )
  (FC_layer): Seq(
    (0): Conv1d(
      (conv): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
      (normlayer): BatchNorm1d(
        (bn): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (activation): ReLU(inplace=True)
    )
    (1): Dropout(p=0.5, inplace=False)
    (2): Conv1d(
      (conv): Conv1d(128, 13, kernel_size=(1,), stride=(1,))
    )
  )
)
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    initial_lr: 0.001
    lr: 0.001
    weight_decay: 0
)
Model size = 1400653
wandb: Tracking run with wandb version 0.8.22
wandb: Run data is saved locally in wandb/run-20200201_174238-gf5g4xcl
wandb: Syncing run sunny-grass-2
wandb: ⭐️ View project at https://app.wandb.ai/thomas-chaton/s3dis
wandb: 🚀 View run at https://app.wandb.ai/thomas-chaton/s3dis/runs/gf5g4xcl
wandb: Run `wandb off` to turn off syncing.

Access tensorboard with the following command <tensorboard --logdir=/home/thoma/deeppointcloud-benchmarks/outputs/2020-02-01/17-42-30/tensorboard>
The provided path pointnet2_charlesssg.pt didn't contain a checkpoint_file
EPOCH 1 / 100
100%|███████████████████████████████████████████████████████████████████████████████████| 1046/1046 [05:13<00:00,  3.33it/s, data_loading=0.255, iteration=0.033, train_acc=73.24, train_loss_seg=1.056, train_macc=55.10, train_miou=39.34)]
Learning rate = 0.001000
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 429/429 [01:21<00:00,  5.27it/s, test_acc=76.92, test_loss_seg=0.929, test_macc=48.98, test_miou=36.93)]
==================================================
    test_loss_seg = 0.9292173981666565
    test_acc = 76.92584267709246
    test_macc = 48.98628983293526
    test_miou = 36.935997572580504
==================================================
EPOCH 2 / 100
100%|███████████████████████████████████████████████████████████████████████████████████| 1046/1046 [05:14<00:00,  3.32it/s, data_loading=0.264, iteration=0.025, train_acc=82.11, train_loss_seg=0.669, train_macc=71.78, train_miou=52.87)]
Learning rate = 0.001000
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 429/429 [01:21<00:00,  5.27it/s, test_acc=77.48, test_loss_seg=0.785, test_macc=54.21, test_miou=39.31)]
==================================================
    test_loss_seg = 0.785090982913971
    test_acc = 77.48870582380144
    test_macc = 54.21480569294253
    test_miou = 39.31108625610043
==================================================
loss_seg: 0.9292173981666565 -> 0.785090982913971, acc: 76.92584267709246 -> 77.48870582380144, macc: 48.98628983293526 -> 54.21480569294253, miou: 36.935997572580504 -> 39.31108625610043
EPOCH 3 / 100
100%|███████████████████████████████████████████████████████████████████████████████████| 1046/1046 [05:14<00:00,  3.32it/s, data_loading=0.256, iteration=0.030, train_acc=84.56, train_loss_seg=0.564, train_macc=76.51, train_miou=57.26)]
Learning rate = 0.001000
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 429/429 [01:21<00:00,  5.29it/s, test_acc=80.08, test_loss_seg=0.618, test_macc=56.31, test_miou=42.04)]
==================================================
    test_loss_seg = 0.6180834770202637
    test_acc = 80.08506939693703
    test_macc = 56.317916729731046
    test_miou = 42.046235768296626
```

