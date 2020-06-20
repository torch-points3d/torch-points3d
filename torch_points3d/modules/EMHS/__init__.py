import sys

from .networks import *
from .modules import *

_custom_models = sys.modules[__name__]


def initialize_emhs(
    model_name,
    input_nc,
    output_nc,
    num_layers,
    module_name,
    num_elm,
    use_attention,
    layers_slice,
    latent_classes,
    voxelization,
    kernel_size,
    feat_dim,
):

    model_cls = getattr(_custom_models, model_name)

    print(
        input_nc, output_nc, num_layers, module_name, layers_slice, latent_classes, voxelization, kernel_size, feat_dim
    )

    return model_cls(
        input_nc=input_nc,
        output_nc=output_nc,
        num_layers=num_layers,
        module_name=module_name,
        num_elm=num_elm,
        use_attention=use_attention,
        layers_slice=layers_slice,
        latent_classes=latent_classes,
        voxelization=voxelization,
        kernel_size=kernel_size,
        feat_dim=feat_dim,
    )
