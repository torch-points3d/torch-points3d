import os
from collections import OrderedDict
from abc import ABC, abstractmethod
from typing import Optional, Dict
import torch
from torch.optim.optimizer import Optimizer
import functools
import operator

class BaseFactory(ABC):
    def __init__(self, module_name_down, module_name_up, modules_lib):
        self.module_name_down = module_name_down
        self.module_name_up = module_name_up
        self.modules_lib = modules_lib

    @abstractmethod
    def get_module_from_index(self):
        pass
    
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
        self.optimizer: Optional[Optimizer] = None
        self._sampling_and_search_dict: Dict = {}

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def get_output(self):
        return self.output

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear existing gradients
        self.backward()              # calculate gradients
        self.optimizer.step()        # update parameters

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_optimizer(self, optimizer_cls: Optimizer, lr=0.001):
        self.optimizer = optimizer_cls(self.parameters(), lr=lr)

    def get_internal_losses(self):
        losses_global = []
        search_key = "internal_losses"
        def search_from_key(modules, losses_global):
            for _, module in modules.items():
                if hasattr(module, search_key):
                    losses_global.append(getattr(module, search_key))   
                search_from_key(module._modules, losses_global)                     
        search_from_key(self._modules, losses_global)
        losses = [[v for v in losses.values()] for losses in losses_global]
        if len(losses) > 0:
            losses = functools.reduce(operator.iconcat, losses, [])
            return torch.mean(torch.stack(losses))
        else:
            return 0.