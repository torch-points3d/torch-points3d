from collections import OrderedDict
from abc import abstractmethod
from typing import Optional, Dict, Any, List
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import logging
from collections import defaultdict
from torch_points3d.core.schedulers.lr_schedulers import instantiate_scheduler
from torch_points3d.core.schedulers.bn_schedulers import instantiate_bn_scheduler
from torch_points3d.utils.enums import SchedulerUpdateOn

from torch_points3d.core.regularizer import *
from torch_points3d.core.losses import instantiate_loss_or_miner
from torch_points3d.utils.config import is_dict
from torch_points3d.utils.colors import colored_print, COLORS
from .model_interface import TrackerInterface, DatasetInterface, CheckpointInterface

log = logging.getLogger(__name__)


class BaseModel(torch.nn.Module, TrackerInterface, DatasetInterface, CheckpointInterface):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    __REQUIRED_DATA__: List[str] = []
    __REQUIRED_LABELS__: List[str] = []

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
        self.visual_names = []
        self.output = None
        self._conv_type = opt.conv_type
        self._optimizer: Optional[Optimizer] = None
        self._lr_scheduler: Optimizer[_LRScheduler] = None
        self._bn_scheduler = None
        self._spatial_ops_dict: Dict = {}
        self._num_epochs = None
        self._num_batches = 0
        self._num_samples = -1
        self._schedulers = {}
        self._accumulated_gradient_step = None
        self._grad_clip = -1
        self._update_lr_scheduler_on = "on_epoch"
        self._update_bn_scheduler_on = "on_epoch"

    @property
    def schedulers(self):
        return self._schedulers

    @schedulers.setter
    def schedulers(self, schedulers):
        if schedulers:
            self._schedulers = schedulers
            for scheduler_name, scheduler in schedulers.items():
                setattr(self, "_{}".format(scheduler_name), scheduler)

    def _add_scheduler(self, scheduler_name, scheduler):
        setattr(self, "_{}".format(scheduler_name), scheduler)
        self._schedulers[scheduler_name] = scheduler

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def num_epochs(self):
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, num_epochs):
        self._num_epochs = num_epochs

    @property
    def num_batches(self):
        return self._num_batches

    @num_batches.setter
    def num_batches(self, num_batches):
        self._num_batches = num_batches

    @property
    def num_samples(self):
        return self._num_samples

    @num_samples.setter
    def num_samples(self, num_samples):
        self._num_samples = num_samples

    @property
    def learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def conv_type(self):
        return self._conv_type

    @conv_type.setter
    def conv_type(self, conv_type):
        self._conv_type = conv_type

    def set_input(self, input, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        raise NotImplementedError

    def get_labels(self):
        """ returns a trensor of size ``[N_points]`` where each value is the label of a point
        """
        return getattr(self, "labels", None)

    def get_batch(self):
        """ returns a trensor of size ``[N_points]`` where each value is the batch index of a point
        """
        return getattr(self, "batch_idx", None)

    def get_output(self):
        """ returns a trensor of size ``[N_points,...]`` where each value is the output
        of the network for a point (output of the last layer in general)
        """
        return self.output

    def get_input(self):
        """ returns the last input that was given to the model or raises error
        """
        return getattr(self, "input")

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        raise NotImplementedError("You must implement your own forward")

    def _manage_optimizer_zero_grad(self):
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

    def _collect_scheduler_step(self, update_scheduler_on):
        if hasattr(self, update_scheduler_on):
            update_scheduler_on = getattr(self, update_scheduler_on)
            if update_scheduler_on is None:
                raise Exception("The function instantiate_optimizers doesn't look like called")

            if update_scheduler_on == SchedulerUpdateOn.ON_EPOCH.value:
                return self._num_epochs
            elif update_scheduler_on == SchedulerUpdateOn.ON_NUM_BATCH.value:
                return self._num_batches
            elif update_scheduler_on == SchedulerUpdateOn.ON_NUM_SAMPLE.value:
                return self._num_samples
        else:
            raise Exception("The attributes {} should be defined within self".format(update_scheduler_on))

    def optimize_parameters(self, epoch, batch_size):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self._num_epochs = epoch
        self._num_batches += 1
        self._num_samples += batch_size

        self.forward(epoch=epoch)  # first call forward to calculate intermediate results
        make_optimizer_step = self._manage_optimizer_zero_grad()  # Accumulate gradient if option is up
        self.backward()  # calculate gradients

        if self._grad_clip > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self._grad_clip)

        if make_optimizer_step:
            self._optimizer.step()  # update parameters

        if self._lr_scheduler:
            lr_scheduler_step = self._collect_scheduler_step("_update_lr_scheduler_on")
            self._lr_scheduler.step(lr_scheduler_step)

        if self._bn_scheduler:
            bn_scheduler_step = self._collect_scheduler_step("_update_bn_scheduler_on")
            self._bn_scheduler.step(bn_scheduler_step)

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

    def instantiate_optimizers(self, config):
        # Optimiser
        optimizer_opt = self.get_from_opt(
            config,
            ["training", "optim", "optimizer"],
            msg_err="optimizer needs to be defined within the training config",
        )
        optmizer_cls_name = optimizer_opt.get("class")
        optimizer_cls = getattr(torch.optim, optmizer_cls_name)
        optimizer_params = {}
        if hasattr(optimizer_opt, "params"):
            optimizer_params = optimizer_opt.params
        self._optimizer = optimizer_cls(self.parameters(), **optimizer_params)

        # LR Scheduler
        scheduler_opt = self.get_from_opt(config, ["training", "optim", "lr_scheduler"])
        if scheduler_opt:
            update_lr_scheduler_on = config.update_lr_scheduler_on
            if update_lr_scheduler_on:
                self._update_lr_scheduler_on = update_lr_scheduler_on
            scheduler_opt.update_scheduler_on = self._update_lr_scheduler_on
            lr_scheduler = instantiate_scheduler(self._optimizer, scheduler_opt)
            self._add_scheduler("lr_scheduler", lr_scheduler)

        # BN Scheduler
        bn_scheduler_opt = self.get_from_opt(config, ["training", "optim", "bn_scheduler"])
        if bn_scheduler_opt:
            update_bn_scheduler_on = config.update_bn_scheduler_on
            if update_bn_scheduler_on:
                self._update_bn_scheduler_on = update_bn_scheduler_on
            bn_scheduler_opt.update_scheduler_on = self._update_bn_scheduler_on
            bn_scheduler = instantiate_bn_scheduler(self, bn_scheduler_opt)
            self._add_scheduler("bn_scheduler", bn_scheduler)

        # Accumulated gradients
        self._accumulated_gradient_step = self.get_from_opt(config, ["training", "optim", "accumulated_gradient"])
        if self._accumulated_gradient_step:
            if self._accumulated_gradient_step > 1:
                self._accumulated_gradient_count = 0
            else:
                raise Exception("When set, accumulated_gradient option should be an integer greater than 1")

        # Gradient clipping
        self._grad_clip = self.get_from_opt(config, ["training", "optim", "grad_clip"], default_value=-1)

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
                        if torch.is_tensor(loss_value):
                            assert loss_value.dim() == 0
                            losses_global[loss_name].append(loss_value)
                        elif isinstance(loss_value, float):
                            losses_global[loss_name].append(torch.tensor(loss_value).to(self.device))
                        else:
                            raise ValueError("Unsupported value type for a loss: {}".format(loss_value))
                search_from_key(module._modules, losses_global)

        search_from_key(self._modules, losses_global)
        return losses_global

    def collect_internal_losses(self, lambda_weight=1, aggr_func=torch.sum):
        """
            Collect internal loss of all child modules with
            internal losses and set the losses
        """
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

    @staticmethod
    def get_metric_loss_and_miner(opt_loss, opt_miner):
        """
        instantiate the loss and the miner if it's available
        in the yaml config:

        example in the yaml config
        metric_loss:
            class: "TripletMarginLoss"
            params:
                smooth_loss: True
                triplets_per_anchors: 'all'
        """
        loss = None
        miner = None
        if opt_loss is not None:
            loss = instantiate_loss_or_miner(opt_loss, mode="metric_loss")
        if opt_miner is not None:
            miner = instantiate_loss_or_miner(opt_miner, mode="miner")

        return loss, miner

    def get_spatial_ops(self):
        return self._spatial_ops_dict

    def enable_dropout_in_eval(self):
        def search_from_key(modules):
            for _, m in modules.items():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()
                search_from_key(m._modules)

        search_from_key(self._modules)

    def get_from_opt(self, opt, keys=[], default_value=None, msg_err=None, silent=True):
        if len(keys) == 0:
            raise Exception("Keys should not be empty")
        value_out = default_value

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
                if not silent:
                    log.exception(e)
            value_out = default_value
        return value_out

    def get_current_visuals(self):
        """Return an OrderedDict containing associated tensors within visual_names"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def log_optimizers(self):
        colored_print(COLORS.Green, "Optimizer: {}".format(self._optimizer))
        colored_print(COLORS.Green, "Learning Rate Scheduler: {}".format(self._lr_scheduler))
        colored_print(COLORS.Green, "BatchNorm Scheduler: {}".format(self._bn_scheduler))
        colored_print(COLORS.Green, "Accumulated gradients: {}".format(self._accumulated_gradient_step))

    def to(self, *args, **kwargs):
        super().to(*args, *kwargs)
        if self.optimizer:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(*args, **kwargs)
        return self

    def verify_data(self, data, forward_only=False):
        """ Goes through the __REQUIRED_DATA__ and __REQUIRED_LABELS__ attribute of the model
        and verifies that the passed data object contains all required members.
        If something is missing it raises a KeyError exception.
        """
        missing_keys = []
        required_attributes = self.__REQUIRED_DATA__
        if not forward_only:
            required_attributes += self.__REQUIRED_LABELS__
        for attr in required_attributes:
            if not hasattr(data, attr) or data[attr] is None:
                missing_keys.append(attr)
        if len(missing_keys):
            raise KeyError(
                "Missing attributes in your data object: {}. The model will fail to forward.".format(missing_keys)
            )

    def print_transforms(self):
        message = ""
        for attr in self.__dict__:
            if "transform" in attr:
                message += "{}{} {}= {}\n".format(COLORS.IPurple, attr, COLORS.END_NO_TOKEN, getattr(self, attr))
        print(message)


class BaseInternalLossModule(torch.nn.Module):
    """ABC for modules which have internal loss(es)
    """

    @abstractmethod
    def get_internal_losses(self) -> Dict[str, Any]:
        pass
