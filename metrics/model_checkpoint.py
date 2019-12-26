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
        self._objects['stats'] = {}
        self._objects['optimizer'] = None
        self._objects['scheduler'] = None
        self._filled = False

    def _load_objects(self):
        try:
            self._objects = torch.load(self._check_path)
            self._filled = True
        except:
            pass

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
                return self._objects['models'][weight_name]
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
            state_dict = self._checkpoint.get_state_dict("best_{}".format(weight_name))
            model._load_from_state_dict(state_dict)
            optimizer = self._checkpoint.get_optimizer()
            model.set_optimizer(optimizer, lr=optimizer.lr)
