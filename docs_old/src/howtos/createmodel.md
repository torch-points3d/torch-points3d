## Create a new model

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
        self._spatial_ops_dict: Dict = {}
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

    def get_spatial_ops(self):
        return self._spatial_ops_dict

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
from src.core.spatial_ops import DenseRadiusNeighbourFinder
from src.core.spatial_ops import DenseFPSSampler
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
