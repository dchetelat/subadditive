import torch

from lp import *
from layers import *


def compute_gomory_round(A, b, c, vtypes, primal_solution, basis, nb_cuts=None):
    M = A[:, basis].inverse()
    u = frac(M@b)

    # Remove cuts with zero u (integer l.h.s.)
    cut_indices = (u > 1e-5) & (u < 1-1e-5)
    M = M[cut_indices, :]
    u = u[cut_indices]

    layer = GomoryLayer(len(b), len(u))
    layer.M.data = M
    layer.v.data = torch.log(u/(1-u))

    # Remove cuts that are not deep enough
    integer_vars = np.isin(vtypes, [GRB.BINARY, GRB.INTEGER])
    cont_vars = np.isin(vtypes, [GRB.CONTINUOUS])
    integral_A = layer(A[:, integer_vars])
    continuous_A = layer.upper(A[:, cont_vars])
    layer_A = torch.cat([integral_A, continuous_A], dim=-1)
    gaps = layer_A@primal_solution - torch.ones_like(u)
    if nb_cuts is None:
        cut_indices = (gaps < 0).nonzero().squeeze(-1)
    else:
        cut_indices = gaps.topk(min(len(gaps), nb_cuts), largest=False).indices
    layer.out_size = len(cut_indices)
    layer.M.data = layer.M.data[cut_indices, :]
    layer.v.data = layer.v.data[cut_indices]
    layer = Cat(layer)

    extended_A, extended_b, extended_c, extended_vtypes = get_extended_lp(A, b, c, vtypes, layer)
    gomory_optimal_value, extended_primal_solution, extended_basis = \
            solve_lp(extended_A, extended_b, extended_c)
    extended_basis = get_basis(extended_A, *extended_basis)

    gomory_info = extended_A, extended_b, extended_c, extended_vtypes, extended_primal_solution, extended_basis
    return gomory_optimal_value, layer.layer.M.data, layer.layer.v.data, gomory_info


def compute_gomory_bounds(A, b, c, vtypes, nb_rounds):
    lp_optimal_value, primal_solution, basis = solve_lp(A, b, c)
    basis = get_basis(A, *basis)

    gomory_bounds = [lp_optimal_value.item()]
    gomory_info = (A, b, c, vtypes, primal_solution, basis)
    for round_ in range(nb_rounds):
        gomory_optimal_value, _, _, gomory_info = compute_gomory_round(*gomory_info)
        gomory_bounds.append(gomory_optimal_value.item())
    return gomory_bounds


def gomory_initialization_(dual_function, A, b, c, vtypes):
    optimal_value, primal_solution, basis = solve_lp(A, b, c)
    basis = get_basis(A, *basis)
    
    gomory_info = (A, b, c, vtypes)
    gomory_solution, gomory_basis = primal_solution, basis

    for layer in dual_function.inner_layers:
        _, gomory_weight, gomory_scale, _ = compute_gomory_round(*gomory_info, gomory_solution, gomory_basis, nb_cuts=layer.layer.M.shape[0])
        if gomory_weight.shape[0] > 0:
            layer.layer.M.data[:gomory_weight.shape[0], :] = gomory_weight.clone().to(layer.layer.M.device)
            layer.layer.v.data[:gomory_weight.shape[0]] = gomory_scale.clone().to(layer.layer.v.device)

        extended_A, extended_b, extended_c, extended_vtypes = get_extended_lp(*gomory_info, layer)
        _, extended_primal_solution, extended_basis = solve_lp(extended_A, extended_b, extended_c)
        extended_basis = get_basis(extended_A, *extended_basis)
        gomory_info = extended_A, extended_b, extended_c, extended_vtypes
        gomory_solution, gomory_basis = extended_primal_solution, extended_basis
