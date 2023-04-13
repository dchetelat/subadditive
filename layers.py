import numpy as np
from gurobipy import GRB
import torch
import torch.nn.functional as F

from utilities import *


class SubadditiveLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._in_size = None
        self._out_size = None
        
    @property
    def in_size(self):
        return self._in_size
    
    @in_size.setter
    def in_size(self, value):
        self._in_size = value
    
    @property
    def out_size(self):
        return self._out_size
    
    @out_size.setter
    def out_size(self, value):
        self._out_size = value
        
    def uddz(self, input_):
        raise NotImplementedError


class GomoryLayer(SubadditiveLayer):
    def __init__(self, in_size, out_size, nonlinear=False):
        super().__init__()
        self.W = torch.nn.Parameter(torch.Tensor(out_size, in_size))
        self.v = torch.nn.Parameter(torch.Tensor(out_size))
        if out_size > 0:
            torch.nn.init.orthogonal_(self.W)
            torch.nn.init.normal_(self.v)
        
        self.in_size = in_size
        self.out_size = out_size
        self.nonlinear = nonlinear
        
    def forward(self, input_):
        v = self.v.sigmoid()
        if self.nonlinear:
            return torch.log(1+weighted_tri(self.W@input_, v)) + weighted_abs(-self.W, v)@input_
        else:
            return weighted_tri(self.W@input_, v) + weighted_abs(-self.W, v)@input_
    
    def uddz(self, input_):
        v = self.v.sigmoid()
        return weighted_abs(self.W@input_, v) + weighted_abs(-self.W, v)@input_


class LinearLayer(SubadditiveLayer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.w = torch.nn.Parameter(torch.Tensor(out_size, in_size))
        torch.nn.init.orthogonal_(self.w)
        self.in_size = in_size
        self.out_size = out_size
    
    def forward(self, input_):
        return self.w@input_
    
    def uddz(self, input_):
        return self.w@input_


class Cat(SubadditiveLayer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        
    @property
    def in_size(self):
        return self.layer.in_size
    
    @property
    def out_size(self):
        return self.layer.in_size + self.layer.out_size
    
    def forward(self, input_):
        return torch.cat([input_, self.layer(input_)], dim=0)
    
    def uddz(self, input_):
        return torch.cat([input_, self.layer.uddz(input_)], dim=0)
    

class SequentialSubadditive(SubadditiveLayer):
    def __init__(self, *arguments):
        super().__init__()
        self.layers = torch.nn.Sequential(*arguments)
        
    @property
    def in_size(self):
        return self.layers[0].in_size
    
    @property
    def out_size(self):
        return self.layers[-1].out_size
        
    def forward(self, input_):
        output = self.layers(input_)
        return output
    
    def uddz(self, input_):
        output = input_
        for layer in self.layers:
            output = layer.uddz(output)
        return output
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return SequentialSubadditive(*self.layers[index])
        else:
            return self.layers[index]
    
    def __len__(self):
        return len(self.layers)

    
def add_cuts_to_ilp(cut_function, A, b, c, vtypes):
    integer_vars = np.isin(vtypes, [GRB.BINARY, GRB.INTEGER])
    continuous_vars = np.isin(vtypes, [GRB.CONTINUOUS])

    extended_A = torch.empty(cut_function.out_size, len(c), device=A.device, dtype=A.dtype)
    extended_A[:, integer_vars] = cut_function(A[:, integer_vars])
    extended_A[:, continuous_vars] = cut_function.uddz(A[:, continuous_vars])
    extended_b = cut_function(b)
    return extended_A, extended_b, c, vtypes

