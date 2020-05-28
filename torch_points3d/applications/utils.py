def extract_output_nc(model_config):
    """ Extracts the number of channels at the output of the network form the model config
    """
    if model_config.up_conv is not None:
        output_nc = model_config.up_conv.up_conv_nn[-1][-1]
    elif model_config.innermost is not None:
        output_nc = model_config.innermost.nn[-1]
    else:
        raise ValueError("Input model_config does not match expected pattern")
    return output_nc
