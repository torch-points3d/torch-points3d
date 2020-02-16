from collections import OrderedDict
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import copy
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import logging
from collections import defaultdict
from src.core.schedulers.lr_schedulers import instantiate_scheduler
from src.core.schedulers.bn_schedulers import instantiate_bn_scheduler

from src.core.regularizer import *
from src.utils.config import is_dict, create_new_omega_conf
from src.utils.metrics_utils import get_from_metric
from src.utils.colors import colored_print, COLORS

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
        self._spatial_ops_dict: Dict = {}
        self._iterations = 0
        self._lr_params = None
        self._grad_clip = getattr(opt, "grad_clip", 0)
        self._latest_metrics = None
        self._latest_stage = None
        self._latest_epoch = None
        self._selection_stage = None
        self._schedulers = {}
        self._model_state = None
        self._accumulated_gradient_step = getattr(opt, "accumulated_gradient", None)
        if self._accumulated_gradient_step:
            if self._accumulated_gradient_step > 1:
                self._accumulated_gradient_count = 0
                colored_print(COLORS.Green, "Accumulated option activated {}".format(self._accumulated_gradient_step))
            else:
                raise Exception("accumulated_gradient should be greater than 1")

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, model_state):
        self._model_state = model_state

    @property
    def lr_params(self):
        try:
            params = copy.deepcopy(self._lr_params)
            params.lr_base = self.learning_rate
            return params
        except:
            return None

    @property
    def schedulers(self):
        return self._schedulers

    @schedulers.setter
    def schedulers(self, schedulers):
        if schedulers:
            self._schedulers = schedulers
            for scheduler_name, scheduler in schedulers.items():
                setattr(self, "_{}".format(scheduler_name), scheduler)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

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

    def set_selection_stage(self, set_selection_stage):
        self._set_selection_stage = set_selection_stage

    def set_metrics(self, metrics, stage, epoch):
        self._latest_metrics = metrics
        self._latest_stage = stage
        self._latest_epoch = epoch

    def manage_optimizer_zero_grad(self):
        if not self._accumulated_gradient_step:
            self._optimizer.zero_grad()  # clear existing gradients
            return True
        else:
            if self._accumulated_gradient_count == self._accumulated_gradient_step:
                self._accumulated_gradient_count = 0
                return True
            if self._accumulated_gradient_count == 0:
                self._optimizer.zero_grad()  # clear existing gradients
            self._accumulated_gradient_count += 1
            return False

    def optimize_parameters(self, batch_size):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self._iterations += batch_size
        self.forward()  # first call forward to calculate intermediate results
        make_optimizer_step = self.manage_optimizer_zero_grad()  # Accumulate gradient if option is up
        self.backward()  # calculate gradients
        if self._grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self._grad_clip)
        if make_optimizer_step:
            self._optimizer.step()  # update parameters
        if self._lr_scheduler is not None:
            if hasattr(self._lr_scheduler, "metric_name"):
                metric_name = self._lr_scheduler
                if hasattr(self, "_metrics"):
                    if self._latest_stage == self._set_selection_stage:
                        metric_value = get_from_metric(self._latest_metrics, metric_name, self._set_selection_stage)
                        if metric_value is None:
                            raise Exception("The provided metric_name in scheduler didn t match any metric_name")
                        log.info("Update lr using metric_value".format(metric_value))
                        self._lr_scheduler.step(metric_value)
            else:
                self._lr_scheduler.step(self._iterations)

        if self._bn_scheduler:
            self._bn_scheduler.step(self._iterations)

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

    def instantiate_optim(self, config):
        optimizer_opt = self.get_from_opt(
            create_new_omega_conf(config.training),
            ["optim", "optimizer"],
            msg_err="optimizer needs to be defined within the training config",
        )

        optmizer_cls_name = optimizer_opt.get("class")
        optimizer_cls = getattr(torch.optim, optmizer_cls_name)
        optimizer_params = {}
        if hasattr(optimizer_opt, "params"):
            optimizer_params = optimizer_opt.params
        self._optimizer = optimizer_cls(self.parameters(), **optimizer_params)
        colored_print(COLORS.Green, "Optimizer: {}".format(self._optimizer))

        scheduler_opt = self.get_from_opt(config, ["training", "optim", "scheduler"])
        if scheduler_opt:
            self._lr_scheduler = instantiate_scheduler(self._optimizer, scheduler_opt)
            self._schedulers["lr_scheduler"] = self._lr_scheduler
            colored_print(COLORS.Green, "Learning Rate Scheduler: {}".format(self._lr_scheduler))

        bn_scheduler_opt = self.get_from_opt(config, ["training", "optim", "bn_scheduler"])
        if bn_scheduler_opt:
            self._bn_scheduler = instantiate_bn_scheduler(self, bn_scheduler_opt, config.training.batch_size)
            self._schedulers["bn_scheduler"] = self._bn_scheduler
            colored_print(COLORS.Green, "BatchNorm Scheduler: {}".format(self._bn_scheduler))

    def get_regularization_loss(self, regularizer_type="L2", **kwargs):
        loss = 0
        regularizer_cls = RegularizerTypes[regularizer_type.upper()].value
        regularizer = regularizer_cls(self, **kwargs)
        return regularizer.regularized_all_param(loss)

    def get_named_internal_losses(self):
        """
            Modules which have internal losses return a dict of the form
            {<loss_name>: <loss>}
            This method merges the dicts of all child modules with internal loss
            and returns this merged dict
        """
        losses_global = defaultdict(list)

        def search_from_key(modules, losses_global):
            for _, module in modules.items():
                if isinstance(module, BaseInternalLossModule):
                    losses = module.get_internal_losses()
                    for loss_name, loss_value in losses.items():
                        losses_global[loss_name].append(loss_value)
                search_from_key(module._modules, losses_global)

        search_from_key(self._modules, losses_global)
        return losses_global

    def collect_internal_losses(self, lambda_weight=1, reduction="sum"):
        """
            Collect internal loss of all child modules with
            internal losses and set the losses
        """
        if reduction == "sum":
            aggr_func = torch.sum

        elif reduction == "mean":
            aggr_func = torch.mean
        else:
            raise NotImplementedError

        loss_out = 0
        losses = self.get_named_internal_losses()
        for loss_name, loss_values in losses.items():
            if loss_name not in self.loss_names:
                self.loss_names.append(loss_name)
            item_loss = lambda_weight * aggr_func(torch.stack(loss_values))
            loss_out += item_loss
            setattr(self, loss_name, item_loss.item())
        return loss_out

    def get_internal_loss(self):
        """
            Returns the average internal loss of all child modules with
            internal losses
        """
        loss = 0
        c = 0
        losses = self.get_named_internal_losses()
        for loss_name, loss_values in losses.items():
            loss += torch.mean(torch.stack(loss_values))
            c += 1
        if c == 0:
            return loss
        else:
            return loss / c

    def get_spatial_ops(self):
        return self._spatial_ops_dict

    def enable_dropout_in_eval(self):
        def search_from_key(modules):
            for _, m in modules.items():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()
                search_from_key(m._modules)

        search_from_key(self._modules)

    def get_from_opt(self, opt, keys=[], msg_err=None):
        if len(keys) == 0:
            raise Exception("Keys should not be empty")

        value_out = None

        def search_with_keys(args, keys, value_out):
            if len(keys) == 0:
                value_out = args
                return value_out
            value = args[keys[0]]
            return search_with_keys(value, keys[1:], value_out)

        try:
            value_out = search_with_keys(opt, keys, value_out)
        except Exception as e:
            if msg_err:
                raise Exception(str(msg_err))
            else:
                print(e)
            value_out = None
        return value_out


class BaseInternalLossModule(ABC):
    """ABC for modules which have internal loss(es)
    """

    @abstractmethod
    def get_internal_losses(self) -> Dict[str, Any]:
        pass
