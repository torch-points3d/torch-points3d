import os
from typing import Dict, Any
import torch
import importlib
from .base_model import BaseModel
from torch_points3d.utils.model_building_utils.model_definition_resolver import resolve_model
from torch_points3d.utils.colors import log_metrics, colored_rank_print
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only


def breakpoint_zero():
    if os.getenv("LOCAL_RANK") == "0":
        import pdb

        pdb.set_trace()


def flatten(data: Any):

    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = flatten(value)

    elif isinstance(data, list):
        if all([torch.is_tensor(value) for value in data]) and len(data) > 0:
            return torch.cat(data, dim=0)

    return data


def instantiate_model(config, dataset) -> BaseModel:
    """ Creates a model given a datset and a training config. The config should contain the following:
    - config.data.task: task that will be evaluated
    - config.model_name: model to instantiate
    - config.models: All models available
    """

    # Get task and model_name
    task = config.data.task
    tested_model_name = config.model_name

    # Find configs
    model_config = getattr(config.models, tested_model_name, None)
    if model_config is None:
        raise Exception("The model_name {} isn t within {}".format(tested_model_name, list(config.models.keys())))
    resolve_model(model_config, dataset, task)

    model_class = getattr(model_config, "class")
    model_paths = model_class.split(".")
    module = ".".join(model_paths[:-1])
    class_name = model_paths[-1]
    model_module = ".".join(["torch_points3d.models", task, module])
    modellib = importlib.import_module(model_module)

    model_cls = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == class_name.lower():
            model_cls = cls

    if model_cls is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (model_module, class_name)
        )
    model = model_cls(model_config, "dummy", dataset, modellib)
    return model


class LitLightningModule(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.tracker_options = {}

    def forward(self, batch, batch_idx):

        self.model.set_input(batch, self.device)
        self.model.forward()

    @property
    def loss(self):
        self.model.compute_loss()
        return self.model.loss

    def step(self, batch, batch_idx, optimizer_idx=None, stage="train"):
        self.forward(batch, batch_idx)
        self.log_metrics(batch, stage)
        return self.loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.step(batch, batch_idx, optimizer_idx, "train")

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        return self.step(batch, batch_idx, optimizer_idx, "val")

    def test_step(self, batch, batch_idx, optimizer_idx=None):
        return self.step(batch, batch_idx, optimizer_idx, "test")

    def reset_tracker(self, stage):
        self.trackers[stage].reset(stage=stage)

    def on_train_epoch_start(self) -> None:
        self.reset_tracker("train")
        self.on_stage_epoch_start("train")
        # self.on_stage_epoch_start("val")

    def on_val_epoch_start(self) -> None:
        self.reset_tracker("val")
        self.on_stage_epoch_start("val")

    def on_test_epoch_start(self) -> None:
        self.reset_tracker("test")
        self.on_stage_epoch_start("test")

    def configure_optimizers(self):
        return [self.model._optimizer]  # , [self.model._schedulers["lr_scheduler"]]

    @staticmethod
    def rename_loss(losses: Dict[torch.Tensor], stage: str = "train"):
        new_losses = dict()
        for key, loss in losses.items():
            if loss is None:
                continue
            loss_key = "%s_%s" % (stage, key)
            new_losses[loss_key] = loss
        return new_losses

    def log_metrics(self, data, stage):
        if not self.trainer.running_sanity_check:
            self.trackers[stage].track(self.model, data=data, **self.tracker_options)
            metrics = self.trackers[stage].get_metrics()
            self.sanetize_metrics(metrics)
            self.log_dict({**metrics}, prog_bar=True, on_step=True, on_epoch=False)

    def on_stage_epoch_start(self, stage):
        tracker = self.trackers[stage]
        metrics = tracker.get_metrics()
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=False)

    def on_train_epoch_end(self, *_) -> None:
        self.on_stage_epoch_end("train")

    def on_validation_epoch_end(self, *_) -> None:
        if not self.trainer.running_sanity_check:
            self.on_stage_epoch_end("val")

    def on_test_epoch_end(self, *_) -> None:
        self.on_stage_epoch_end("test")

    def on_stage_epoch_end(self, stage):
        os.getenv("LOCAL_RANK", None)
        tracker = self.trackers[stage]

        # is_dist_initialized = torch.distributed.is_available() and torch.distributed.is_initialized()

        # if is_dist_initialized:

        #     def convert_numpy(data, dtype=np.float):
        #         return data.cpu().numpy()

        #     for key, value in vars(tracker).items():
        #         if isinstance(value, (tuple, dict, list, torch.Tensor)):
        #             new_value = flatten(self.all_gather(value))
        #             new_value = apply_to_collection(new_value, torch.Tensor, convert_numpy)
        #             # colored_rank_print(f"\n {rank} {key} \n {value} \n {new_value} \n")
        #             setattr(tracker, key, new_value)
        tracker.finalise()
        metrics = tracker.get_metrics()
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        self.log_metrics_epoch_end(metrics, stage)
        self.reset_tracker(stage)

    @rank_zero_only
    def log_metrics_epoch_end(self, metrics, stage):
        log_metrics(metrics, stage)

    def sanetize_metrics(self, metrics):
        try:
            for loss_name in self.model.loss_names:
                for key in metrics.keys():
                    if loss_name in key:
                        del metrics[key]
        except:
            pass


def convert_to_lightning_module(model: BaseModel) -> LitLightningModule:
    return LitLightningModule(model)
