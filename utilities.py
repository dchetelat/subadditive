import torch


def weighted_tri(x, scale=None, scale_cmp=None):
    if scale is None:
        scale = torch.DoubleTensor([0.5]).squeeze().to(x.device)
    scale = expand_dims(scale, x)

    return torch.min(frac(x)/scale, (1-frac(x))/(1-scale))


def weighted_abs(x, scale=None, scale_cmp=None):
    if scale is None:
        scale = torch.DoubleTensor([0.5]).squeeze().to(x.device)
    scale = expand_dims(scale, x)

    return torch.max(x/scale, -x/(1-scale))


def expand_dims(x, y):
    x = x.view(list(x.shape) + [1 for _ in range(y.dim()-x.dim())])
    return x.expand(y.shape)


def frac(input_, tolerance=0):
    return input_-(input_+tolerance).floor()
