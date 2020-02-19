import os
import torch
import logging
import copy
from omegaconf import OmegaConf, DictConfig
from src.models.base_model import BaseModel
from src.utils.colors import COLORS, colored_print
from src.core.schedulers.lr_schedulers import instantiate_scheduler
from src.core.schedulers.bn_schedulers import instantiate_bn_scheduler
from src import instantiate_model

log = logging.getLogger(__name__)

DEFAULT_METRICS_FUNC = {
    "iou": max,
    "acc": max,
    "loss": min,
    "mer": min,
}  # Those map subsentences to their optimization functions


class Checkpoint:
    _LATEST = "latest"

    def __init__(self, checkpoint_file: str, save_every_iter: bool = True):
        """ Checkpoint manager. Saves to working directory with check_name
        Arguments
            checkpoint_file {str} -- Path to the checkpoint
            save_every_iter {bool} -- [description] (default: {True})
        """
        self._check_path = checkpoint_file
        self._filled = False
        self.run_config = None
        self.models = {}
        self.stats = {"train": [], "test": [], "val": []}
        self.optimizer = None
        self.schedulers = {}

    def save_objects(self, models_to_save, stage, current_stat, optimizer, schedulers, **kwargs):
        """ Saves checkpoint with updated mdoels for the given stage
        """
        self.models = models_to_save
        self.stats[stage].append(current_stat)
        self.optimizer = [optimizer.__class__.__name__, optimizer.state_dict()]
        self.schedulers = {
            scheduler_name: [scheduler.scheduler_opt, scheduler.state_dict()]
            for scheduler_name, scheduler in schedulers.items()
        }

        to_save = kwargs
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                to_save[key] = value
        torch.save(to_save, self._check_path)

    @staticmethod
    def load(checkpoint_dir: str, checkpoint_name: str, run_config: DictConfig):
        """ Creates a new checkpoint object in the current working directory by loading the
        checkpoint located at [checkpointdir]/[checkpoint_name].pt
        """
        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name) + ".pt"
        ckp = Checkpoint(checkpoint_file)
        if not os.path.exists(checkpoint_file):
            log.warning("The provided path {} didn't contain a checkpoint_file".format(checkpoint_file))
            ckp.run_config = run_config
            return ckp
        log.info("Loading checkpoint from {}".format(checkpoint_file))
        objects = torch.load(checkpoint_file)
        for key, value in objects.items():
            setattr(ckp, key, value)
        ckp._filled = True
        return ckp

    @property
    def is_empty(self):
        return not self._filled

    def get_optimizer(self, params):
        if not self.is_empty:
            try:
                optimizer_config = self.optimizer
                optimizer_cls = getattr(torch.optim, optimizer_config[0])
                optimizer = optimizer_cls(params)
                optimizer.load_state_dict(optimizer_config[1])
                return optimizer
            except:
                raise KeyError("The checkpoint doesn t contain an optimizer")

    def get_schedulers(self, model):
        if not self.is_empty:
            try:
                schedulers_out = {}
                schedulers_config = self.schedulers
                for scheduler_type, (scheduler_opt, scheduler_state) in schedulers_config.items():
                    if scheduler_type == "lr_scheduler":
                        optimizer = model.optimizer
                        scheduler = instantiate_scheduler(optimizer, scheduler_opt)
                        scheduler.load_state_dict(scheduler_state)
                        schedulers_out["lr_scheduler"] = scheduler
                    elif scheduler_type == "bn_scheduler":
                        scheduler = instantiate_bn_scheduler(model, scheduler_opt)
                        scheduler.load_state_dict(scheduler_state)
                        schedulers_out["bn_scheduler"] = scheduler
                    else:
                        raise NotImplementedError
                return schedulers_out
            except:
                log.warn("The checkpoint doesn t contain schedulers")
                return None

    def get_state_dict(self, weight_name):
        if not self.is_empty:
            try:
                models = self.models
                keys = [key.replace("best_", "") for key in models.keys()]
                log.info("Available weights : {}".format(keys))
                try:
                    key_name = "best_{}".format(weight_name)
                    model = models[key_name]
                    log.info("Model loaded from {}:{}".format(self._check_path, key_name))
                    return model
                except:
                    key_name = Checkpoint._LATEST
                    model = models[Checkpoint._LATEST]
                    log.info("Model loaded from {}:{}".format(self._check_path, key_name))
                    return model
            except:
                raise Exception("This weight name isn't within the checkpoint ")


class ModelCheckpoint(object):
    """ Create a checkpoint for a given model

    Argumemnts:
        - load_dir: directory where to load the checkpoint from (if exists)
        -
    """

    def __init__(self, load_dir: str, check_name: str, resume: bool, selection_stage: str, run_config: DictConfig):
        self._checkpoint = Checkpoint.load(load_dir, check_name, copy.deepcopy(run_config))
        self._resume = resume
        self._selection_stage = selection_stage

    def create_model(self, dataset, weight_name=Checkpoint._LATEST):
        if not self.is_empty:
            run_config = copy.deepcopy(self._checkpoint.run_config)
            model = instantiate_model(run_config, dataset)
            self._initialize_model(model, weight_name)
            return model
        else:
            raise ValueError("Checkpoint is empty")

    @property
    def start_epoch(self):
        if self._resume:
            return self.get_starting_epoch()
        else:
            return 1

    @property
    def selection_stage(self):
        return self._selection_stage

    @selection_stage.setter
    def selection_stage(self, value):
        self._selection_stage = value

    @property
    def is_empty(self):
        return self._checkpoint.is_empty

    def get_starting_epoch(self):
        return len(self._checkpoint.stats["train"]) + 1

    def _initialize_model(self, model: BaseModel, weight_name):
        if not self._checkpoint.is_empty:
            state_dict = self._checkpoint.get_state_dict(weight_name)
            model.load_state_dict(state_dict)
            if self._resume:
                model.optimizer = self._checkpoint.get_optimizer(model.parameters())
                model.schedulers = self._checkpoint.get_schedulers(model)

    def find_func_from_metric_name(self, metric_name, default_metrics_func):
        for token_name, func in default_metrics_func.items():
            if token_name in metric_name:
                return func
        raise Exception(
            'The metric name {} doesn t have a func to measure which one is best. Example: For best_train_iou, {"iou":max}'.format(
                token_name
            )
        )

    def save_best_models_under_current_metrics(self, model: BaseModel, metrics_holder: dict, **kwargs):
        """[This function is responsible to save checkpoint under the current metrics and their associated DEFAULT_METRICS_FUNC]
        Arguments:
            model {[BaseModel]} -- [Model]
            metrics_holder {[Dict]} -- [Need to contain stage, epoch, current_metrics]
        """
        metrics = metrics_holder["current_metrics"]
        stage = metrics_holder["stage"]
        epoch = metrics_holder["epoch"]

        stats = self._checkpoint.stats
        state_dict = model.state_dict()

        current_stat = {}
        current_stat["epoch"] = epoch

        models_to_save = self._checkpoint.models

        if stage == "train":
            models_to_save[Checkpoint._LATEST] = state_dict

        if len(stats[stage]) > 0:
            latest_stats = stats[stage][-1]

            msg = ""
            improved_metric = 0

            for metric_name, current_metric_value in metrics.items():
                current_stat[metric_name] = current_metric_value

                metric_func = self.find_func_from_metric_name(metric_name, DEFAULT_METRICS_FUNC)
                best_metric_from_stats = latest_stats["best_{}".format(metric_name)]
                best_value = metric_func(best_metric_from_stats, current_metric_value)
                current_stat["best_{}".format(metric_name)] = best_value

                # This new value seems to be better under metric_func
                if (self._selection_stage == stage) and (
                    current_metric_value == best_value
                ):  # Update the model weights
                    models_to_save["best_{}".format(metric_name)] = state_dict

                    msg += "{}: {} -> {}, ".format(metric_name, best_metric_from_stats, best_value)
                    improved_metric += 1

            if improved_metric > 0:
                colored_print(COLORS.VAL_COLOR, msg[:-2])
        else:
            # stats[stage] is empty.
            for metric_name, metric_value in metrics.items():
                current_stat[metric_name] = metric_value
                current_stat["best_{}".format(metric_name)] = metric_value
                models_to_save["best_{}".format(metric_name)] = state_dict

        self._checkpoint.save_objects(models_to_save, stage, current_stat, model.optimizer, model.schedulers, **kwargs)
