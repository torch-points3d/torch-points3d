from torch_points3d.models.base_model import BaseModel
import torch
from torch_points3d.core.schedulers.lr_schedulers import instantiate_scheduler
from torch_points3d.core.schedulers.bn_schedulers import instantiate_bn_scheduler


class BaseGanModel(BaseModel):
    
    def __init__(self, option):
        super(BaseGanModel, self).__init__(option)

        self.scales = option.scales
        self.latent_space = option.latent_space

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
        self.g_optimizer = self._optimizer
        self.d_optimizers = []
        for i in range(len(self.scales)):
            self.d_optimizers.append(optimizer_cls(self.parameters(), **optimizer_params))

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

    def optimize_parameters(self, epoch, batch_size):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self._num_epochs = epoch
        self._num_batches += 1
        self._num_samples += batch_size

        self.d_step(epoch=epoch)
        self.g_step(epoch=epoch)

    def g_step(self, *args, **kwargs):
        raise NotImplementedError("You must implement your own g_step")

    def d_step(self, *args, **kwargs):
        raise NotImplementedError("You must implement your own d_step")
