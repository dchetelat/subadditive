import numpy as np
import gurobipy as gp
from gurobipy import GRB
import torch


def get_condition_number(matrix):
    epsilon = (torch.solve(matrix, matrix)[0] - torch.eye(matrix.shape[0]).to(matrix.device)).abs().max()
    return epsilon


def inv_softplus(x, tolerance=0):
    return (x.exp()-1+tolerance).log()


# class Frac(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, tolerance):
#         return x-(x+tolerance).floor()

#     @staticmethod
#     def backward(ctx, grad_output):
#         # pretend the frac was really 0.5*sin(2*pi*(x+0.5))+0.5, 
#         # which has derivative pi*cos(2*pi(x+0.5))
#         if ctx.needs_input_grad[0]:
#             grad_input = np.pi*torch.cos(2*np.pi*(grad_output+0.5))*grad_output
#             return grad_input, None
#         else:
#             return None, None
# frac = Frac.apply


def frac(input_, tolerance=0):
    return input_-(input_+tolerance).floor()

def expand_dims(x, y):
    x = x.view(list(x.shape) + [1 for _ in range(y.dim()-x.dim())])
    return x.expand(y.shape)

def weighted_tri(x, scale=None, scale_cmp=None):
    if scale is None:
        scale = torch.FloatTensor([0.5]).squeeze().to(x.device)
    if scale_cmp is None:
        scale_cmp = 1 - scale
    scale = expand_dims(scale, x)
    scale_cmp = expand_dims(scale_cmp, x)

    y = torch.zeros_like(x)
    y[x>0] = torch.min(x.frac()/scale, (1-x.frac())/scale_cmp)[x>0]
    y[x<0] = torch.min(x.abs().frac()/scale_cmp, (1-x.abs().frac())/scale)[x<0]
    return y

def weighted_abs(x, scale=None, scale_cmp=None):
    if scale is None:
        scale = torch.FloatTensor([0.5]).squeeze().to(x.device)
    if scale_cmp is None:
        scale_cmp = 1 - scale
    scale = expand_dims(scale, x)
    scale_cmp = expand_dims(scale_cmp, x)

    y = torch.zeros_like(x)
    y[x>0] = (x/scale)[x>0]
    y[x<0] = (x.abs()/scale_cmp)[x<0]
    return y

def oneinv(x):
    y = torch.ones_like(x)
    y[x!=0] = 1/x[x!=0]
    return y


class SparseCat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, dim):
        z = torch.cat([x, y], dim)
        ctx.save_for_backward(torch.LongTensor([x.shape[dim]]).squeeze(),
                              torch.LongTensor([dim]).squeeze())
        return z
    
    @staticmethod
    def backward(ctx, gradient):
        x_shape, dim = ctx.saved_tensors
        
        gradient = gradient.coalesce()
        indices, values = gradient.indices(), gradient.values()

        x_indices = indices[dim, :] < x_shape
        y_indices = indices[dim, :] >= x_shape
        x_values = values[x_indices]
        y_values = values[y_indices]
        x_indices = indices[:, x_indices]
        y_indices = indices[:, y_indices]
        y_indices[dim, :] -= x_shape

        dimensions = list(gradient.shape)
        dimensions[dim] = x_shape
        gradient_x = torch.sparse_coo_tensor(x_indices, x_values, dimensions)
        dimensions[dim] = gradient.shape[dim]-x_shape
        gradient_y = torch.sparse_coo_tensor(y_indices, y_values, dimensions)
        
        return gradient_x, gradient_y, None
sparse_cat = SparseCat.apply
