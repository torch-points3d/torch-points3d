from enum import Enum
from typing import *
from omegaconf import DictConfig
import logging
import importlib
from torch_geometric.transforms import Compose
from torch_points3d.core.data_transform import XYZFeature, AddFeatsByKeys

log = logging.getLogger(__name__)


class ModelArchitectures(Enum):
    UNET = "unet"
    ENCODER = "encoder"
    DECODER = "decoder"


def get_module_lib(module_name):
    model_module = ".".join(["torch_points3d.modules", module_name])
    return importlib.import_module(model_module)


class ModelFactory:

    MODEL_ARCHITECTURES = [e.value for e in ModelArchitectures]

    @staticmethod
    def raise_enum_error(arg_name, arg_value, options):
        raise Exception("The provided argument {} with value {} isn't within {}".format(arg_name, arg_value, options))

    def __init__(
        self,
        architecture: str = None,
        input_nc: int = None,
        output_nc: int = None,
        num_layers: int = None,
        channel_nn: List[int] = None,
        weights: str = False,
        use_rgb: bool = False,
        use_normal: bool = False,
        use_z: bool = False,
        config: DictConfig = None,
        **kwargs
    ):

        self._architecture = architecture.lower()
        assert self._architecture in self.MODEL_ARCHITECTURES, ModelFactory.raise_enum_error(
            "model_architecture", self._architecture, self.MODEL_ARCHITECTURES
        )
        self._input_nc = input_nc
        self._output_nc = output_nc
        self._num_layers = num_layers
        self._channel_nn = channel_nn
        self._weights = weights
        self._config = config
        self._use_rgb = use_rgb
        self._use_normal = use_normal
        self._use_z = use_z
        self._kwargs = kwargs

        if self._config:
            log.info("The config will be used to build the model")

        self._modellib = get_module_lib(self.MODULE_NAME)

        self._transform = self._build_transform()

        print(self._transform)

    def _check_init_transform(self):
        cnd_1 = hasattr(self, "_transforms")
        cnd_2 = hasattr(self, "_list_add_to_x")
        hasattr(self, "_feat_names")
        cnd_4 = hasattr(self, "_delete_feats")

        if not cnd_1:
            self._transforms = []

        if not (cnd_1 and cnd_2 and cnd_4):
            self._list_add_to_x = []
            self._feat_names = []
            self._input_nc_feats = []
            self._delete_feats = []

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def num_features(self):
        raise NotImplementedError

    def _build_transform(self):
        if self._use_z:
            self._transforms.append(XYZFeature(add_x=False, add_y=False, add_z=True))

        if self._use_rgb:
            self._list_add_to_x += [True]
            self._feat_names += ["rgb"]
            self._input_nc_feats += [3]
            self._delete_feats += [True]

        if self._use_normal:
            self._list_add_to_x += [True]
            self._feat_names += ["normal"]
            self._input_nc_feats += [3]
            self._delete_feats += [True]

        self._transforms.append(
            AddFeatsByKeys(
                list_add_to_x=self._list_add_to_x,
                feat_names=self._feat_names,
                input_nc_feats=self._input_nc_feats,
                delete_feats=self._delete_feats,
            )
        )

        return Compose(self._transforms)

    def _build_unet(self):
        return self._modellib.build_unet(self)

    def _build_encoder(self):
        return self._modellib.build_encoder(self)

    def _build_decoder(self):
        return self._modellib.build_decoder(self)

    def _build(self):
        if self._architecture == ModelArchitectures.UNET.value:
            self._build_unet()
        elif self._architecture == ModelArchitectures.ENCODER.value:
            self._build_encoder()
        elif self._architecture == ModelArchitectures.DECODER.value:
            self._build_decoder()
        else:
            raise NotImplementedError
