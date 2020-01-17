import os
from collections import OrderedDict, ChainMap
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import copy
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import functools
import operator
import logging

from schedulers.lr_schedulers import get_scheduler

log = logging.getLogger(__name__)


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
        pass

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
        pass

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


class BaseInternalLossModule(ABC):
    """ABC for modules which have internal loss(es)
    """

    @abstractmethod
    def get_internal_losses(self) -> Dict[str, Any]:
        pass
