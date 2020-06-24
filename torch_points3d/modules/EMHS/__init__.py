import sys

from .networks import *
from .modules import *

_custom_models = sys.modules[__name__]


def initialize_emhs(
    model_name,
    input_nc,
    output_nc,
    num_layers,
    num_elm,
    use_attention,
    layers_slice,
    latent_classes,
    voxelization,
    kernel_size,
    feat_dim,
    attention_type,
):

    print(
        {
            "model_name": model_name,
            "input_nc": input_nc,
            "output_nc": output_nc,
            "num_layers": num_layers,
            "num_elm": num_elm,
            "use_attention": use_attention,
            "layers_slice": layers_slice,
            "latent_classes": latent_classes,
            "voxelization": voxelization,
            "kernel_size": kernel_size,
            "feat_dim": feat_dim,
            "attention_type": attention_type,
        }
    )

    model_cls = getattr(_custom_models, model_name)

    return model_cls(
        input_nc=input_nc,
        output_nc=output_nc,
        num_layers=num_layers,
        num_elm=num_elm,
        use_attention=use_attention,
        layers_slice=layers_slice,
        latent_classes=latent_classes,
        voxelization=voxelization,
        kernel_size=kernel_size,
        feat_dim=feat_dim,
        attention_type=attention_type,
    )
