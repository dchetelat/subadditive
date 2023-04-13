import torch
from lp import *
from layers import *
from utilities import *


def compute_gomory_bounds(A, b, c, vtypes, nb_rounds, nonlinear=False):
    gomory_bounds = []
    for _ in range(nb_rounds+1):
        W, v, gomory_bound = compute_gomory_weights(A, b, c, vtypes)
        gomory_bounds.append(gomory_bound)

        layer = Cat(GomoryLayer(len(b), len(v), nonlinear=nonlinear))
        layer.layer.W.data = W
        layer.layer.v.data = v

        A, b, c, vtypes = add_cuts_to_ilp(layer, A, b, c, vtypes)
    return gomory_bounds


def gomory_initialization_(dual_function, A, b, c, vtypes):
    for layer in dual_function.inner_layers:
        W, v, _ = compute_gomory_weights(A, b, c, vtypes)
        nb_cuts = min(layer.layer.out_size, len(v))

        if nb_cuts > 0:
            layer.layer.W.data[:nb_cuts, :] = W[:nb_cuts, :].clone().to(layer.layer.W.device)
            layer.layer.v.data[:nb_cuts] = v[:nb_cuts].clone().to(layer.layer.v.device)

        A, b, c, vtypes = add_cuts_to_ilp(layer, A, b, c, vtypes)


def compute_gomory_weights(extended_A, extended_b, c, vtypes):
    cuts_lp_value, _, basis = solve_lp(extended_A, extended_b, c)
    basis = get_joint_basis(extended_A, *basis)
    slack_A = -torch.eye(extended_A.shape[0], device=extended_A.device)
    extended_B = torch.cat([extended_A, slack_A], dim=1)[:, basis]

    W = extended_B.inverse()
    u = frac(W@extended_b)

    # Remove cuts with zero u (integer l.h.s.)
    cut_indices = (u > 1e-5) & (u < 1-1e-5)
    W, u = W[cut_indices, :], u[cut_indices]
    
    v = torch.log(u/(1-u))
    return W, v, cuts_lp_value
