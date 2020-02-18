import os
import torch
import logging
from omegaconf import OmegaConf, ListConfig, DictConfig
from src.models.base_model import BaseModel
from src.utils.colors import COLORS, colored_print
from src.core.schedulers.lr_schedulers import instantiate_scheduler
from src.core.schedulers.bn_schedulers import instantiate_bn_scheduler
from src import instantiate_model
from src.utils.model_building_utils.model_definition_resolver import resolve_model

log = logging.getLogger(__name__)

DEFAULT_METRICS_FUNC = {
    "iou": max,
    "acc": max,
    "loss": min,
    "mer": min,
}  # Those map subsentences to their optimization functions


class Checkpoint(object):
    _LATEST = "latest"

    def __init__(self, checkpoint_dir: str, save_every_iter: bool = True):
        """ Checkpoint manager. Saves to working directory with check_name.
        Assumes that the working directory is an output folder of Hydra
        If the model file already exists then it loads its state
        Arguments
            checkpoint_file {str} -- Path to the checkpoint
            save_every_iter {bool} -- [description] (default: {True})
        """
        try:
            self._config_dict = OmegaConf.load(os.path.join(checkpoint_dir, ".hydra/config.yaml"))
            self._run_config = self._parse_command_line(
                OmegaConf.load(os.path.join(checkpoint_dir, ".hydra/overrides.yaml"))
            )
        except:
            raise ValueError("The checkpoint directory %s does not seem do be a hydra directory" % checkpoint_dir)

        checkpoint_file = os.path.join(checkpoint_dir, self.model_name) + ".pt"
        self._check_path = checkpoint_file
        self._initialize_objects()

        if not os.path.exists(checkpoint_file):
            log.warning("The provided path {} didn't contain a checkpoint_file".format(checkpoint_file))
        else:
            log.info("Loading checkpoint from {}".format(checkpoint_file))
            self._objects = torch.load(checkpoint_file)

        self._filled = True

    @staticmethod
    def _parse_command_line(cmd_arguments: ListConfig):
        """ Parses the command line arguments (list of arg=value string) returns a dict_config
        """
        config = {}
        for arg in cmd_arguments:
            try:
                key, value = arg.split("=")
                config[key] = value
            except:
                pass
        return DictConfig(config)

    def _initialize_objects(self):
        self._objects = {}
        self._objects["models"] = {}
        self._objects["stats"] = {"train": [], "test": [], "val": []}
        self._objects["optimizer"] = None
        self._filled = False

    def save_objects(self, models_to_save, stage, current_stat, optimizer, schedulers, **kwargs):
        self._objects["models"] = models_to_save
        self._objects["stats"][stage].append(current_stat)
        self._objects["optimizer"] = [optimizer.__class__.__name__, optimizer.state_dict()]
        schedulers_saved = {
            scheduler_name: [scheduler.scheduler_opt, scheduler.state_dict()]
            for scheduler_name, scheduler in schedulers.items()
        }
        self._objects["schedulers"] = schedulers_saved
        torch.save(self._objects, self._check_path)

    @property
    def models_to_save(self):
        return self._objects["models"]

    @property
    def stats(self):
        return self._objects["stats"]

    @property
    def is_empty(self):
        return not self._filled

    @property
    def task(self):
        return self._run_config.task

    @property
    def model_name(self):
        return self._run_config.model_name

    @property
    def model_config(self):
        return self._config_dict.models[self.model_name]

    @property
    def config(self):
        return self._config_dict

    @property
    def data_config(self):
        return self._config_dict.data

    @property
    def training_config(self):
        return self._config_dict.training

    @property
    def conv_type(self):
        return self.model_config.conv_type

    def get_optimizer(self, params):
        if not self.is_empty:
            try:
                optimizer_config = self._objects["optimizer"]
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
                schedulers_config = self._objects["schedulers"]
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
                models = self._objects["models"]
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


class ModelCheckpoint(Checkpoint):
    def __init__(self, load_dir: str = None, resume: bool = True, selection_stage: str = "test"):
        super().__init__(load_dir)
        self._resume = resume
        self._selection_stage = selection_stage

    def create_model_from_checkpoint(self, dataset, weight_name=None):
        if not self.is_empty:
            model_name = self.model_name
            model_config = getattr(self.config.models, model_name, None)
            model_class = getattr(model_config, "class")
            task = self.task
            resolve_model(model_config, dataset, task)
            option = OmegaConf.merge(model_config, self.config.training)
            model = instantiate_model(model_class, task, option, dataset)
            if weight_name:
                self._initialize_model(model)
            return model

    @property
    def start_epoch(self):
        if self._resume:
            return self.get_starting_epoch()
        else:
            return 1

    def get_starting_epoch(self):
        return len(self.stats["train"]) + 1

    def _initialize_model(self, model: BaseModel, weight_name: str = None):
        if not self.is_empty:
            state_dict = self.get_state_dict(weight_name)
            model.load_state_dict(state_dict)
            if self._resume:
                model.optimizer = self.get_optimizer(model.parameters())
                model.schedulers = self.get_schedulers(model)

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

        stats = self.stats
        state_dict = model.state_dict()

        current_stat = {}
        current_stat["epoch"] = epoch

        models_to_save = self.models_to_save

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

        self.save_objects(models_to_save, stage, current_stat, model.optimizer, model.schedulers, **kwargs)
