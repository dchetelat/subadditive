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

def triangle_wave(input_, tolerance=0):
    output = frac(input_, tolerance)
    return torch.min(output, 1-output)

def oneinv(x):
    y = torch.ones_like(x)
    y[x!=0] = 1/x[x!=0]
    return y


