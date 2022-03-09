import numpy as np
import torch
import torch.nn.functional as F

from utilities import *
from lp import *


class SubadditiveLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_size = None
        self.out_size = None
        
    def upper(self, input_):
        raise NotImplementedError
        
    def final_fit(self):
        raise NotImplementedError
        
    def needs_bounding(self):
        raise NotImplementedError


class GomoryLayer(SubadditiveLayer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.M = torch.nn.Parameter(torch.Tensor(out_size, in_size))
        self.v = torch.nn.Parameter(torch.Tensor(out_size))
        if out_size > 0:
            torch.nn.init.orthogonal_(self.M, gain=0.1)
            torch.nn.init.normal_(self.v)
        
        self.in_size = in_size
        self.out_size = out_size
        
    def forward(self, input_):
        hidden = self.M@input_
        scale = self.v.sigmoid()
        scale_cmp = 1/(1+self.v.exp())
        
        return weighted_tri(hidden, scale, scale_cmp)
    
    def upper(self, input_):
        hidden = self.M@input_
        scale = self.v.sigmoid()
        scale_cmp = 1/(1+self.v.exp())
        
        return weighted_abs(hidden, scale, scale_cmp)
    
    def needs_bounding(self):
        return torch.BoolTensor([True for _ in range(self.out_size)])
    

class SparseGomoryLayer(SubadditiveLayer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.M = torch.nn.Parameter(torch.Tensor(out_size, in_size))
        self.v = torch.nn.Parameter(torch.Tensor(out_size))
        torch.nn.init.orthogonal_(self.M)
        torch.nn.init.normal_(self.v)
        
        self.in_size = in_size
        self.out_size = out_size
        
    def forward(self, input_):
        if input_.is_sparse:
            hidden = torch.sparse.mm(input_.t(), self.M.t()).t()
        else:
            hidden = self.M@input_
        scale = self.v.sigmoid()
        scale_cmp = 1/(1+self.v.exp())
#         if hidden.dim() == 2:
#             scale = scale.unsqueeze(-1).expand(hidden.shape)
#             scale_cmp = scale_cmp.unsqueeze(-1).expand(hidden.shape)
        
#         output = torch.min(hidden/scale, (1-hidden)/scale_cmp)
        output = weighted_tri(hidden, scale, scale_cmp)
        return output.to_sparse() if input_.is_sparse else output
        
    def upper(self, input_):
        if input_.is_sparse:
            hidden = torch.sparse.mm(input_.t(), self.M.t()).t()
        else:
            hidden = self.M@input_
        scale = self.v.sigmoid()
        scale_cmp = 1/(1+self.v.exp())
#         if hidden.dim() == 2:
#             scale = scale.unsqueeze(-1).expand(hidden.shape)
#             scale_cmp = scale_cmp.unsqueeze(-1).expand(hidden.shape)
        
#         output = torch.zeros_like(hidden)
#         output[hidden < 0] = (-hidden/scale_cmp)[hidden < 0]
#         output[hidden > 0] = (hidden/scale)[hidden > 0]

        output = weighted_abs(hidden, scale, scale_cmp)
        return output.to_sparse() if input_.is_sparse else output
    
    def needs_bounding(self):
        return torch.BoolTensor([True for _ in range(self.out_size)])


class LinearLayer(SubadditiveLayer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.w = torch.nn.Parameter(torch.Tensor(out_size, in_size))
        torch.nn.init.orthogonal_(self.w)
        self.in_size = in_size
        self.out_size = out_size
    
    def forward(self, input_):
        return self.w@input_
    
    def upper(self, input_):
        return self.w@input_
    
    def needs_bounding(self):
        return torch.BoolTensor([False for _ in range(self.out_size)])
        
    def final_fit(self, loss_A, loss_b, loss_c, basis_start=None):
        assert self.out_size == 1
        optimal_value, primal_solution, dual_solution, primal_basis, dual_basis = solve_lp(loss_A, loss_b, loss_c, basis_start=basis_start)
        self.w.data = dual_solution.unsqueeze(0)
        
        return optimal_value, primal_solution, dual_solution, primal_basis, dual_basis


class Cat(SubadditiveLayer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.in_size = self.layer.in_size
        self.out_size = self.layer.in_size + self.layer.out_size
    
    def forward(self, input_):
        return torch.cat([input_, self.layer(input_)], dim=0)
    
    def upper(self, input_):
        return torch.cat([input_, self.layer.upper(input_)], dim=0)
    
    def needs_bounding(self):
        return torch.cat([
            torch.BoolTensor([False for _ in range(self.in_size)]),
            self.layer.needs_bounding()
        ])

    
class SparseCat(SubadditiveLayer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.in_size = self.layer.in_size
        self.out_size = self.layer.in_size + self.layer.out_size
    
    def forward(self, input_):
        if input_.is_sparse:
            return sparse_cat(input_, self.layer(input_), 0)
        else:
            return torch.cat([input_, self.layer(input_)], dim=0)
    
    def upper(self, input_):
        if input_.is_sparse:
            return sparse_cat(input_, self.layer.upper(input_), 0)
        else:
            return torch.cat([input_, self.layer.upper(input_)], dim=0)
    
    def needs_bounding(self):
        return torch.cat([
            torch.BoolTensor([False for _ in range(self.in_size)]),
            self.layer.needs_bounding()
        ])


class SequentialSubadditive(SubadditiveLayer):
    def __init__(self, *arguments):
        super().__init__()
        self.layers = torch.nn.Sequential(*arguments)
        
        self.in_size = arguments[0].in_size
        self.out_size = arguments[-1].out_size
        
    def forward(self, input_):
        output = self.layers(input_)
        return output
    
    def upper(self, input_):
        output = input_
        for layer in self.layers:
            output = layer.upper(output)
        return output
    
    def needs_bounding(self):
        return self.layer[-1].needs_bounding()
    
    def __getitem__(self, index):
        return self.layers[index]
    
    def __len__(self):
        return len(self.layers)


def get_extended_lp(A, b, c, vtypes, layer):
    device = A.device
    integral_vars = np.isin(vtypes, [GRB.BINARY, GRB.INTEGER])
    continuous_vars = np.isin(vtypes, [GRB.CONTINUOUS])

    integral_A = layer(A[:, integral_vars])
    continuous_A = layer.upper(A[:, continuous_vars])
    slack_A = -torch.eye(layer.out_size, device=device)[:, layer.needs_bounding()]
    extended_A = torch.cat([integral_A, continuous_A, slack_A], axis=1)
    extended_b = layer(b)
    extended_c = torch.cat([c[integral_vars], c[continuous_vars], 
                            torch.zeros(slack_A.shape[1], device=device)], dim=0)
    extended_vtypes = np.concatenate([vtypes, [GRB.CONTINUOUS for _ in range(slack_A.shape[1])]])
    
    return extended_A, extended_b, extended_c, extended_vtypes


def get_sparse_extended_lp(integral_A, continuous_A, b, integral_c, continuous_c, vtypes, layer):
    device = integral_A.device
    integral_A = layer(integral_A)
    continuous_A = layer.upper(continuous_A)
    slack_A = -torch.eye(layer.out_size, device=device)[:, layer.needs_bounding()].to(device).to_sparse()
    
    extended_integral_A = integral_A
    extended_continuous_A = sparse_cat(continuous_A, slack_A, 1)
    extended_b = layer(b)
    extended_integral_c = integral_c
    extended_continuous_c = torch.cat([continuous_c, torch.zeros(slack_A.shape[1], device=device)], dim=0)
    extended_vtypes = np.concatenate([vtypes, [GRB.CONTINUOUS for _ in range(slack_A.shape[1])]])

    return extended_integral_A, extended_continuous_A, extended_b, extended_integral_c, extended_continuous_c, extended_vtypes
