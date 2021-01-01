import importlib
import torch
from .base_model import BaseModel
from torch_points3d.utils.model_building_utils.model_definition_resolver import resolve_model
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule


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
        self.tracker = None
        self.tracker_options = {}

    def forward(self, batch, batch_idx):
        self.model.set_input(batch, self.device)
        self.model.forward()

    @property
    def loss(self):
        losses = []
        for loss_name in self.model.loss_names:
            losses.append(getattr(self.model, loss_name, None))
        if len(losses) > 1:
            raise NotImplementedError
        return losses[0]

    def step(self, batch, batch_idx, optimizer_idx = None, stage = "train"):
        self.forward(batch, batch_idx)
        self.log_metrics(batch, stage)
        if stage == "train":
            return self.loss

    def training_step(self, batch, batch_idx, optimizer_idx = None):
        return self.step(batch, batch_idx, optimizer_idx, "train")

    def validation_step(self, batch, batch_idx, optimizer_idx = None):
        return self.step(batch, batch_idx, optimizer_idx, "val")

    def test_step(self, batch, batch_idx, optimizer_idx = None):
        return self.step(batch, batch_idx, optimizer_idx, "test")

    def log_metrics(self, data, stage):
        if self.tracker is not None:
            self.tracker.track(self.model, data=data, **self.tracker_options)
            metrics = self.tracker.get_metrics()
            try:
                for loss_name in self.model.loss_names:
                    del metrics[f"{stage}_{loss_name}"]
            except:
                pass
            self.log_dict(metrics, prog_bar=True, on_step = True, on_epoch=False)

    def reset_tracker(self):
        self.stage = self.trainer.logger_connector._current_stage.value
        self.tracker = self.trackers[self.stage]
        self.tracker.reset(stage=self.stage)        

    def on_train_epoch_start(self) -> None:
        self.reset_tracker()

    def on_val_epoch_start(self) -> None:
        self.reset_tracker()

    def on_test_epoch_start(self) -> None:
        self.reset_tracker()

    def on_epoch_end(self) -> None:
        for key, value in vars(self.tracker).items():
            setattr(self, key, self.all_gather(value))
        metrics = self.tracker.publish(self.trainer.current_epoch)
        for key, value in metrics.items():
            print(f"{key}: {value}")

    def configure_optimizers(self):
        return [self.model._optimizer] #, [self.model._schedulers["lr_scheduler"]]


def convert_to_lightning_module(model: BaseModel) -> LitLightningModule:
    return LitLightningModule(model)
