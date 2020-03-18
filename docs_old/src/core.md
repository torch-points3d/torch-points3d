# Base Conv

The ```/src/core/base_conv folder``` implements base convolution helpers for all ConvType.

<h4> BaseConvolution </h4>

```python
from abc import ABC
import numpy as np
import torch

class BaseConvolution(ABC, torch.nn.Module):
    def __init__(self, sampler, neighbour_finder, *args, **kwargs):
        torch.nn.Module.__init__(self)

        self.sampler = sampler
        self.neighbour_finder = neighbour_finder

    @property
    def nb_params(self):
        """[This property is used to return the number of trainable parameters for a given layer]
        It is useful for debugging and reproducibility.
        Returns:
            [type] -- [description]
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self._nb_params = sum([np.prod(p.size()) for p in model_parameters])
        return self._nb_params
```

This ```BaseConvolution``` set the ```sampler``` and ```neighbour_finder``` used within the ```forward method```.

<h4> DENSE ConvType Format </h4>

It contains the followings class:

* BaseDenseConvolutionDown
* BaseDenseConvolutionUp
* DenseFPModule
* GlobalDenseBaseModule


One should implement the conv function with those parameters
```python
* BaseDenseConvolutionDown
    def conv(self, x, pos, new_pos, radius_idx, scale_idx):
        """ Implements a Dense convolution where radius_idx represents
        the indexes of the points in x and pos to be agragated into the new feature
        for each point in new_pos

        Arguments:
            x -- Previous features [B, C, N]
            pos -- Previous positions [B, N, 3]
            new_pos  -- Sampled positions [B, npoints, 3]
            radius_idx -- Indexes to group [B, npoints, nsample]
            scale_idx -- Scale index in multiscale convolutional layers
        """
        raise NotImplementedError
```

One should implement the conv function with those parameters
```python
* BaseDenseConvolutionUp
    def conv(self, pos, pos_skip, x):
        raise NotImplementedError

```

Here is an example for ```DenseFPModule``` using ```BaseDenseConvolutionUp```

```python
class DenseFPModule(BaseDenseConvolutionUp):
    def __init__(self, up_conv_nn, bn=True, bias=False, activation="LeakyReLU", **kwargs):
        super(DenseFPModule, self).__init__(None, **kwargs)

        self.nn = pt_utils.SharedMLP(up_conv_nn, bn=bn, activation=get_activation(activation))

    def conv(self, pos, pos_skip, x):
        assert pos_skip.shape[2] == 3

        if pos is not None:
            dist, idx = tp.three_nn(pos_skip, pos)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = tp.three_interpolate(x, idx, weight)
        else:
            interpolated_feats = x.expand(*(x.size()[0:2] + (pos_skip.size(1),)))

        return interpolated_feats

    def __repr__(self):
        return "{}: {} ({})".format(self.__class__.__name__, self.nb_params, self.nn)
```

```python
* GlobalDenseBaseModule:
    def forward(self, data):
        x, pos = data.x, data.pos
        pos_flipped = pos.transpose(1, 2).contiguous()

        x = self.nn(torch.cat([x, pos_flipped], dim=1).unsqueeze(-1))

        if self._aggr == "max":
            x = x.squeeze(-1).max(-1)[0]
        elif self._aggr == "mean":
            x = x.squeeze(-1).mean(-1)
        else:
            raise NotImplementedError("The following aggregation {} is not recognized".format(self._aggr))

        pos = None  # pos.mean(1).unsqueeze(1)
        x = x.unsqueeze(-1)
        return Data(x=x, pos=pos)
```

<h4> MESSAGE_PASSING ConvType Format </h4>

It contains the followings class:

* BaseConvolutionDown
* BaseMSConvolutionDown
* BaseConvolutionUp
* GlobalBaseModule
* BaseResnetBlockDown
* BaseResnetBlock

```python
* BaseConvolutionDown
    def conv(self, x, pos, edge_index, batch):
        raise NotImplementedError
```

```python
* BaseConvolutionUp
    def conv(self, x, pos, pos_skip, batch, batch_skip,             edge_index):
        raise NotImplementedError
```

<h4> PARTIAL_DENSE ConvType Format </h4>

It contains the followings class:

* BasePartialDenseConvolutionDown
* GlobalPartialDenseBaseModule

```python
* BasePartialDenseConvolutionDown
    def conv(self, x, pos, x_neighbour,  pos_centered_neighbour, idx_neighbour, idx_sampler):
        """ Generic down convolution for partial dense data

        Arguments:
            x [N, C] -- features
            pos [N, 3] -- positions
            x_neighbour [N, n_neighbours, C] -- features of the neighbours of each point in x
            pos_centered_neighbour [N, n_neighbours, 3]  -- position of the neighbours of x_i centred on x_i
            idx_neighbour [N, n_neighbours] -- indices of the neighbours of each point in pos
            idx_sampler [n]  -- points to keep for the output

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError
```

# Common Modules

<h4> PARTIAL_DENSE ConvType Format </h4>

* MLP
* Identity
* UnaryConv

<h4> Spatial Transformer </h4>

* BaseLinearTransformSTNkD

# Data Transform

* GridSampling
* Center
* RandomTranslate
* RandomScale
* RandomSymmetry
* RandomNoise
* RandomRotation
* MeshToNormal
* MultiScaleTransform

# Sampling

The sampler are built using the ```BaseSampler``` class.

```python
class BaseSampler(ABC):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def __init__(self, ratio=None, num_to_sample=None, subsampling_param=None):
        if num_to_sample is not None:
            if (ratio is not None) or (subsampling_param is not None):
                raise ValueError("Can only specify ratio or num_to_sample or subsampling_param, not several !")
            self._num_to_sample = num_to_sample

        elif ratio is not None:
            self._ratio = ratio

        elif subsampling_param is not None:
            self._subsampling_param = subsampling_param

        else:
            raise Exception('At least ["ratio, num_to_sample, subsampling_param"] should be defined')

    def __call__(self, pos, x=None, batch=None):
        return self.sample(pos, batch=batch, x=x)

    def _get_num_to_sample(self, batch_size) -> int:
        if hasattr(self, "_num_to_sample"):
            return self._num_to_sample
        else:
            return math.floor(batch_size * self._ratio)

    def _get_ratio_to_sample(self, batch_size) -> float:
        if hasattr(self, "_ratio"):
            return self._ratio
        else:
            return self._num_to_sample / float(batch_size)

    @abstractmethod
    def sample(self, pos, x=None, batch=None):
        pass
```

* FPSSampler
* GridSampler
* RandomSampler

* DenseFPSSampler
* DenseRandomSampler

# Neighbour Finder

The sampler are built either using the ```BaseNeighbourFinder``` class or the ```BaseMSNeighbourFinder``` (MS stands for multi-scale)

```python
class BaseNeighbourFinder(ABC):
    def __call__(self, x, y, batch_x, batch_y):
        return self.find_neighbours(x, y, batch_x, batch_y)

    @abstractmethod
    def find_neighbours(self, x, y, batch_x, batch_y):
        pass

    def __repr__(self):
        return str(self.__class__.__name__) + " " + str(self.__dict__)
```

or 
```python
class BaseMSNeighbourFinder(ABC):
    def __call__(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        return self.find_neighbours(x, y, batch_x=batch_x, batch_y=batch_y, scale_idx=scale_idx)

    @abstractmethod
    def find_neighbours(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        pass

    @property
    @abstractmethod
    def num_scales(self):
        pass
```

* RadiusNeighbourFinder
* KNNNeighbourFinder
* DilatedKNNNeighbourFinder

* MultiscaleRadiusNeighbourFinder
* DenseRadiusNeighbourFinder

# Schedulers

* step_decay