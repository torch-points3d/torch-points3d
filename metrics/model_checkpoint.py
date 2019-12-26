import os
import tempfile
import warnings
import sys
import torch

from models.base_model import BaseModel


class Checkpoint(object):

    def __init__(self, to_save: str = None, check_name: str = None, save_every_iter: bool = True):
        if not os.path.exists(to_save):
            os.makedirs(to_save)

        self._to_save = to_save
        self._check_name = check_name
        self._check_path = os.path.join(to_save, "{}.p".format(check_name))
        self._initialize_objects()
        self._load_objects()

    def _initialize_objects(self):
        self._objects = {}
        self._objects['models'] = {}
        self._objects['stats'] = {'train': [], 'test': [], 'val': []}
        self._objects['optimizer'] = None
        self._objects['scheduler'] = None
        self._objects['args'] = None
        self._objects['kwargs'] = None
        self._filled = False

    def save_objects(self, models_to_save, stage, current_stat, optimizer, scheduler, **kwargs):
        print(models_to_save.keys())
        self._objects['models'] = models_to_save
        self._objects['stats'][stage].append(current_stat)
        self._objects['optimizer'] = optimizer
        self._objects['scheduler'] = scheduler
        #self._objects['kwargs'] = kwargs
        torch.save(self._objects, self._check_path)

    def _load_objects(self):
        try:
            self._objects = torch.load(self._check_path)
            self._filled = True
        except:
            pass

    @property
    def models_to_save(self):
        return self._objects['models']

    @property
    def stats(self):
        return self._objects['stats']

    @property
    def is_empty(self):
        return not self._filled

    def get_optimizer(self):
        if not self.is_empty:
            try:
                return self._objects['optimizer']
            except:
                raise Exception("The checkpoint doesn t contain an optimizer")

    def get_state_dict(self, weight_name):
        if not self.is_empty:
            try:
                models = self._objects['models']
                try:
                    key_name = "best_train_{}".format(weight_name)
                    model = models["best_train_{}".format(weight_name)]
                    print("Model loaded from {}/{}".format(self._check_path, key_name))
                    return model
                except:
                    key_name = 'default'
                    model = models['default']
                    print("Model loaded from {}/{}".format(self._check_path, key_name))
                    return model
            except:
                raise Exception("This weight name isn't within the checkpoint ")

    @staticmethod
    def load_objects(to_save: str = None, check_name: str = None):
        return Checkpoint(to_save, check_name)


class ModelCheckpoint(object):

    def __init__(self, to_save: str = None, check_name: str = None):
        self._checkpoint = Checkpoint.load_objects(to_save, check_name)

    def initialize_model(self, model: BaseModel, weight_name: str = None):
        if not self._checkpoint.is_empty:
            state_dict = self._checkpoint.get_state_dict(weight_name)
            model.load_state_dict(state_dict)
            optimizer = self._checkpoint.get_optimizer()
            model.set_optimizer(optimizer.__class__, lr=optimizer.defaults['lr'])

    def find_func_from_metric_name(self, metric_name, default_metrics_func):
        for token_name, func in default_metrics_func.items():
            if token_name in metric_name:
                return func
        raise Exception(
            'The metric name doesn t have a func to measure which one is best. Example: For best_train_iou, {"iou":max}')

    def save_object(self, kwargs, stage, n_iter, metrics, default_metrics_func):

        stats = self._checkpoint.stats
        model = kwargs.get('model')
        state_dict = model.state_dict()
        optimizer = model.optimizer

        current_stat = {}
        current_stat['epoch'] = n_iter

        models_to_save = self._checkpoint.models_to_save
        models_to_save['default'] = state_dict

        if len(stats[stage]) > 0:
            latest_stats = stats[stage][-1]

            for metric_name, metric_value in metrics.items():
                current_stat[metric_name] = metric_value

                metric_func = self.find_func_from_metric_name(metric_name, default_metrics_func)
                best_metric = latest_stats['best_{}'.format(metric_name)]
                best_value = metric_func(best_metric, metric_value)
                current_stat['best_{}'.format(metric_name)] = best_value

                # This new value seems to be better under metric_func
                if (("test" in metric_name) and best_metric != best_value):  # Update the model weights
                    import pdb
                    pdb.set_trace()
                    models_to_save['best_{}'.format(metric_name)] = state_dict
        else:
            # Stats are empty.
            models_to_save = {}
            for metric_name, metric_value in metrics.items():
                current_stat[metric_name] = metric_value
                current_stat['best_{}'.format(metric_name)] = metric_value
                if stage == "test":
                    models_to_save['best_{}'.format(metric_name)] = model.state_dict()

        self._checkpoint.save_objects(models_to_save, stage, current_stat, optimizer, None, **kwargs)
