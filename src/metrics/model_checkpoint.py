import os
import torch
import logging
from omegaconf import OmegaConf
from src.models.base_model import BaseModel
from src.utils.colors import COLORS, colored_print
from src.core.schedulers.lr_schedulers import instantiate_scheduler
from src import instantiate_model

log = logging.getLogger(__name__)

DEFAULT_METRICS_FUNC = {
    "iou": max,
    "acc": max,
    "loss": min,
    "mer": min,
}  # Those map subsentences to their optimization functions


def get_model_checkpoint(
    model: BaseModel,
    load_dir: str,
    check_name: str,
    resume: bool = True,
    weight_name: str = None,
    selection_stage: str = "test",
):
    """ Loads a model from a checkpoint or creates a new one.
    """
    model.set_selection_stage(selection_stage)

    model_checkpoint: ModelCheckpoint = ModelCheckpoint(load_dir, check_name, resume, selection_stage)

    if resume:
        model_checkpoint.initialize_model(model, weight_name)
    return model_checkpoint


class Checkpoint(object):
    _LATEST = "latest"

    def __init__(self, checkpoint_file: str, save_every_iter: bool = True):
        """ Checkpoint manager. Saves to working directory with check_name
        Arguments
            checkpoint_file {str} -- Path to the checkpoint
            save_every_iter {bool} -- [description] (default: {True})
        """
        self._check_path = checkpoint_file
        self._initialize_objects()

    def _initialize_objects(self):
        self._objects = {}
        self._objects["models"] = {}
        self._objects["model_state"] = None
        self._objects["stats"] = {"train": [], "test": [], "val": []}
        self._objects["optimizer"] = None
        self._objects["lr_params"] = None
        self._filled = False

    def save_objects(self, models_to_save, model_state, stage, current_stat, optimizer, schedulers, **kwargs):
        self._objects["models"] = models_to_save
        self._objects["model_state"] = model_state
        self._objects["stats"][stage].append(current_stat)
        self._objects["optimizer"] = [optimizer.__class__.__name__, optimizer.state_dict()]
        schedulers_saved = {scheduler_name: [scheduler.scheduler_opt, scheduler.state_dict()]
                for scheduler_name, scheduler in schedulers.items()}
        self._objects["schedulers"] = schedulers_saved
        colored_print(COLORS.Green, "Saving checkpoint at {}".format(self._check_path))
        torch.save(self._objects, self._check_path)

    @staticmethod
    def load(checkpoint_dir: str, checkpoint_name: str):
        """ Creates a new checpoint object in the current working directory by loading the
        checkpoint located at [checkpointdir]/[checkpoint_name].pt
        """
        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name) + ".pt"
        ckp = Checkpoint(checkpoint_file)
        if not os.path.exists(checkpoint_file):
            log.warn("The provided path {} didn't contain a checkpoint_file".format(checkpoint_file))
            return ckp
        log.info("Loading checkpoint from {}".format(checkpoint_file))
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

    def get_model_state(self):
        if not self.is_empty:
            try:
                return self._objects["model_state"]
            except:
                raise KeyError("The checkpoint doesn t contain model_state")

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

    def get_schedulers(self, optimizer):
        if not self.is_empty:
            try:
                schedulers_out = {}
                schedulers_config = self._objects["schedulers"]
                for scheduler_type, (scheduler_opt, scheduler_state) in schedulers_config.items():
                    if scheduler_type == "lr_scheduler":
                        scheduler = instantiate_scheduler(optimizer, scheduler_opt)
                        scheduler.load_state_dict(scheduler_state)
                        schedulers_out["lr_scheduler"] = scheduler
                    else:
                        raise NotImplementedError
                return schedulers_out
            except:
                log.warn("The checkpoint doesn t contain schedulers")
                return None

    def get_lr_params(self):
        if not self.is_empty:
            try:
                return self._objects["lr_params"]
            except:
                params = build_basic_params()
                log.warning(
                    "Could not find learning rate parameters in this checkpoint, takes the default ones {}".format(
                        params
                    )
                )
                return params

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


class ModelCheckpoint(object):
    def __init__(
        self, load_dir: str = None, check_name: str = None, resume: bool = True, selection_stage: str = "test"
    ):
        self._checkpoint = Checkpoint.load(load_dir, check_name)
        self._resume = resume
        self._selection_stage = selection_stage

    def create_model_from_checkpoint(self, dataset, weight_name=None):
        if not self._checkpoint.is_empty:
            model_state = self._checkpoint.get_model_state()
            model_class = model_state["model_class"]
            option = OmegaConf.create(model_state["option"])
            import pdb; pdb.set_trace()
            task = model_state["dataset_state"]["task"]
            model = instantiate_model(model_class, task, \
                option, dataset)
            
            if weight_name:
                self.initialize_model(model)

            return model

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
            model.model_state = self._checkpoint.get_model_state()
            model.optimizer = self._checkpoint.get_optimizer(model.parameters())
            model.schedulers = self._checkpoint.get_schedulers(model.optimizer)

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

        model.set_metrics(metrics, stage, epoch)

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

        self._checkpoint.save_objects(models_to_save, model.model_state, stage, current_stat, model.optimizer, model.schedulers, **kwargs)
