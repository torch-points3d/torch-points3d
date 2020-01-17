import os
from os import path as osp
import tempfile
import warnings
import sys
import torch
import logging

from models.base_model import BaseModel
from metrics.colored_tqdm import COLORS
from utils_folder.utils import colored_print
from schedulers.lr_schedulers import build_basic_params

log = logging.getLogger(__name__)

DEFAULT_METRICS_FUNC = {
    "iou": max,
    "acc": max,
    "loss": min,
}  # Those map subsentences to their optimization functions


def get_model_checkpoint(
    model: BaseModel, load_dir: str, check_name: str, resume: bool = True, weight_name: str = None,
):
    """ Loads a model from a checkpoint or creates a new one.
    """
    model_checkpoint: ModelCheckpoint = ModelCheckpoint(load_dir, check_name, resume)

    if resume:
        model_checkpoint.initialize_model(model, weight_name)
    return model_checkpoint


class Checkpoint(object):
    _LATEST = "latest"

    def __init__(self, check_name: str, save_every_iter: bool = True):
        """ Checkpoint manager. Saves to working directory with check_name

        Arguments
            check_name {str} -- name of the checkpoint
            save_every_iter {bool} -- [description] (default: {True})
        """
        self._check_path = "{}.pt".format(check_name)
        self._initialize_objects()

    def _initialize_objects(self):
        self._objects = {}
        self._objects["models"] = {}
        self._objects["stats"] = {"train": [], "test": [], "val": []}
        self._objects["optimizer"] = None
        self._objects["lr_params"] = None
        self._filled = False

    def save_objects(self, models_to_save, stage, current_stat, optimizer, lr_params, **kwargs):
        self._objects["models"] = models_to_save
        self._objects["stats"][stage].append(current_stat)
        self._objects["optimizer"] = optimizer
        self._objects["lr_params"] = lr_params
        torch.save(self._objects, self._check_path)

    @staticmethod
    def load(checkpoint_dir: str, checkpoint_name: str):
        """ Creates a new checpoint object in the current working directory by loading the
        checkpoint located at [checkpointdir]/[checkpoint_name].pt
        """
        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name) + ".pt"
        if not os.path.exists(checkpoint_file):
            raise ValueError("Checkpoint %s does not exists" % checkpoint_file)
        ckp = Checkpoint(checkpoint_name)
        ckp._objects = torch.load(checkpoint_file)
        ckp._filled = True
        return ckp

    @property
    def models_to_save(self):
        return self._objects["models"]

    @property
    def stats(self):
        return self._objects["stats"]

    @property
    def is_empty(self):
        return not self._filled

    def get_optimizer(self):
        if not self.is_empty:
            try:
                return self._objects["optimizer"]
            except:
                raise KeyError("The checkpoint doesn t contain an optimizer")

    def get_lr_params(self):
        if not self.is_empty:
            try:
                return self._objects["lr_params"]
            except:
                params = build_basic_params()
                log.warning(
                    "Could not find learning rate parameters in teyh checkpoint, takes the default ones {}".format(
                        params
                    )
                )
                return params

    def get_state_dict(self, weight_name):
        if not self.is_empty:
            try:
                models = self._objects["models"]
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
    def __init__(self, load_dir: str = None, check_name: str = None, resume: bool = True):
        self._checkpoint = Checkpoint.load(load_dir, check_name)
        self._resume = resume

    @property
    def start_epoch(self):
        if self._resume:
            return self.get_starting_epoch()
        else:
            return 1

    def get_starting_epoch(self):
        return len(self._checkpoint.stats["train"]) + 1

    def initialize_model(self, model: BaseModel, weight_name: str = None):
        if not self._checkpoint.is_empty:
            state_dict = self._checkpoint.get_state_dict(weight_name)
            model.load_state_dict(state_dict)
            optimizer = self._checkpoint.get_optimizer()
            lr_params = self._checkpoint.get_lr_params()
            model.set_optimizer(optimizer.__class__, lr_params=lr_params)

    def find_func_from_metric_name(self, metric_name, default_metrics_func):
        for token_name, func in default_metrics_func.items():
            if token_name in metric_name:
                return func
        raise Exception(
            'The metric name doesn t have a func to measure which one is best. Example: For best_train_iou, {"iou":max}'
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

        models_to_save = self._checkpoint.models_to_save

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
                if ("test" == stage) and (current_metric_value == best_value):  # Update the model weights
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

        self._checkpoint.save_objects(models_to_save, stage, current_stat, model.optimizer, model.lr_params, **kwargs)
