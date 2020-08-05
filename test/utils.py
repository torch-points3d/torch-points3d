import torch


def test_hasgrad(model, strict=False, verbose=False):
    """ Tests if a pytorch module has got parameters with gradient equal to 0. Returns the
    ratio of parameters with 0 gradient against the total number of parameters

    Parameters
    ----------
    strict:
        If True then raises an error
    verbose:
        If True then displays all the parameter's names that have 0 gradient
    """
    count = 0
    total_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            total_params += 1
            assert p.grad is not None
            if torch.nonzero(p.grad, as_tuple=False).sum() == 0:
                if verbose:
                    print("Param with name %s has 0 grad" % name)
                count += 1
    if count > 0:
        msg = "Model has %.2f%% of parameters with 0 gradient" % (count / (1.0 * total_params))
        if strict:
            raise ValueError(msg)
        else:
            return count / (1.0 * total_params)
    return 1
