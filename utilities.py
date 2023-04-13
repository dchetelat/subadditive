import torch


def weighted_tri(x, scale=None):
    if scale is None:
        scale = torch.ones(x.shape, device=x.device)
    scale = expand_dims(scale, x)
    
    return torch.minimum(frac(x)/scale, (1-frac(x))/(1-scale))


def weighted_abs(x, scale=None):
    if scale is None:
        scale = torch.ones(x.shape, device=x.device)
    scale = expand_dims(scale, x)

    return torch.maximum(x/scale, -x/(1-scale))


def expand_dims(x, y):
    x = x.view(list(x.shape) + [1 for _ in range(y.dim()-x.dim())])
    return x.expand(y.shape)


def frac(input_, tolerance=0):
    return input_-(input_+tolerance).floor()
