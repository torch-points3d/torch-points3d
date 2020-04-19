from enum import Enum
from typing import *
from omegaconf import DictConfig
import logging
import importlib


log = logging.getLogger(__name__)


class ModelArchitectures(Enum):
    UNET = "unet"
    ENCODER = "encoder"
    DECODER = "decoder"


class ModelTypes(Enum):
    KPCONV = "KPConv"
    MINKOWSKI_ENGINE = "MinkowskiEngine"
    POINTCNN = "PointCNN"
    POINTNET = "pointnet"
    POINTNET2 = "pointnet2"
    RSCONV = "RSConv"
    SPARSECONV = "sparseconv"


def create_model(
    model_type: str = None,
    model_architecture: str = None,
    input_nc: int = None,
    output_nc: int = None,
    channel_nn: List[int] = None,
    pre_trained: bool = False,
    config: DictConfig = None,
    **kwargs
):

    model_factory = ModelFactory(
        model_type, model_architecture, input_nc, output_nc, channel_nn, pre_trained, config, **kwargs
    )
    return model_factory.build()


class ModelFactory:

    MODEL_TYPES = [e.name.lower() for e in ModelTypes]
    MODEL_TYPES_MODULES = [e.value for e in ModelTypes]
    MODEL_ARCHITECTURES = [e.value for e in ModelArchitectures]

    @staticmethod
    def raise_enum_error(arg_name, arg_value, options):
        raise Exception("The provided argument {} with value {} isn't within {}".format(arg_name, arg_value, options))

    def __init__(
        self,
        model_type: str = None,
        model_architecture: str = None,
        input_nc: int = None,
        output_nc: int = None,
        channel_nn: List[int] = None,
        pre_trained: bool = False,
        config: DictConfig = None,
        **kwargs
    ):

        self._model_type = model_type.lower()
        self._model_architecture = model_architecture.lower()
        assert self._model_type in self.MODEL_TYPES, ModelFactory.raise_enum_error(
            "model_type", self._model_type, self.MODEL_TYPES
        )
        assert self._model_architecture in self.MODEL_ARCHITECTURES, ModelFactory.raise_enum_error(
            "model_architecture", self._model_architecture, self.MODEL_ARCHITECTURES
        )
        self._input_nc = input_nc
        self._ioutput_nc = output_nc
        self._channel_nn = channel_nn
        self._pre_trained = pre_trained
        self._config = config
        self._kwargs = kwargs

        if self._config:
            log.info("The config will be used to build the model")

        idx_model_type = self.MODEL_TYPES.index(self._model_type)
        model_module = ".".join(["torch_points3d.modules", self.MODEL_TYPES_MODULES[idx_model_type]])
        self._modellib = importlib.import_module(model_module)

    def _build_unet(self):
        return self._modellib.build_unet(self)

    def _build_encoder(self):
        return self._modellib.build_encoder(self)

    def _build_decoder(self):
        return self._modellib.build_decoder(self)

    def build(self):
        if self._model_architecture == ModelArchitectures.UNET.value:
            self._build_unet()
        elif self._model_architecture == ModelArchitectures.ENCODER.value:
            self._build_encoder()
        elif self._model_architecture == ModelArchitectures.DECODER.value:
            self._build_decoder()
        else:
            raise NotImplementedError
