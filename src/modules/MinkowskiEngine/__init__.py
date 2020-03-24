import sys
from .networks import *
from .unet import UnwrappedSparseUnet

_custom_models = sys.modules[__name__]

def initialize_custom_minkowski_unet(
    input_nc,
    output_nc,
    input_planes=[8, 16, 32, 64, 128],
    norm_layer=ME.MinkowskiBatchNorm,
    use_dropout=False,
    n_reps=1,
    dim=3,
    dropout_rate=0.1,
    mix_conv=False,
):

    return UnwrappedSparseUnet(
        input_nc,
        output_nc,
        input_planes=input_planes,
        norm_layer=norm_layer,
        use_dropout=use_dropout,
        n_reps=n_reps,
        dim=dim,
        dropout_rate=dropout_rate,
        mix_conv=mix_conv,
    )


def initialize_baseline_minkowski_unet(model_name, in_channels, out_channels, D):
    net_cls = getattr(_custom_models, model_name)
    return net_cls(in_channels=in_channels, out_channels=out_channels, D=D)
