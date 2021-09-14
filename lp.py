import numpy as np
import gurobipy as gp
from gurobipy import GRB
import torch


def load_instance(instance_path, device="cpu", add_variable_bounds=True, presolve=False):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        
        model = gp.read(instance_path, env=env)
        if presolve:
            model = model.presolve()

        constraints, variables = model.getConstrs(), model.getVars()
        m, n = len(constraints), len(variables)
        A, b, c = torch.zeros(m, n), torch.zeros(m), torch.zeros(n)

        A = torch.FloatTensor(model.getA().todense())

        for constraint_index, constraint in enumerate(constraints):
            b[constraint_index] = constraint.RHS

        for variable_index, variable in enumerate(variables):
            c[variable_index] = model.ModelSense*variable.Obj

        vtypes = [variable.VType for variable in variables]

        # Add slack variables
        slack_A, slack_c, slack_vtypes = [], [], []
        for constraint_index in range(len(constraints)):
            sense = constraints[constraint_index].Sense
            if sense != '=':
                a = torch.zeros(len(b), 1)
                a[constraint_index, 0] = 1 if sense == '<' else -1
                slack_A.append(a)
                slack_c.append(torch.zeros(1))
                slack_vtypes.append(GRB.CONTINUOUS)
        A = torch.cat([A, *slack_A], dim=-1)
        c = torch.cat([c, *slack_c])
        vtypes += slack_vtypes

        # Add variable upper bounds
        if add_variable_bounds:
            bound_A = torch.cat([torch.eye(len(variables)), 
                                 torch.zeros(len(variables), A.shape[1]-len(variables)), 
                                 torch.eye(len(variables))], dim=1)
            A = torch.cat([A, torch.zeros(A.shape[0], len(variables))], dim=1)
            A = torch.cat([A, bound_A], dim=0)
            b = torch.cat([b, torch.FloatTensor([variable.UB for variable in variables])])
            c = torch.cat([c, torch.zeros(len(variables))])
            vtypes += [GRB.CONTINUOUS for _ in range(len(variables))]

        for variable in variables:
            assert variable.LB == 0

    return A.to(device), b.to(device), c.to(device), vtypes


class LinearProgram(torch.autograd.Function):
    """
    N.B. Only differentiable with respect to optimal value.
    """
    @staticmethod
    def forward(ctx, A, b, c, basis_start=None, point_start=None, verbose=False, method='simplex'):
        device = A.device
        A, b, c = A.detach().cpu().numpy(), b.detach().cpu().numpy(), c.detach().cpu().numpy()
        
        with gp.Env(empty=True) as env:
            if not verbose:
                env.setParam('OutputFlag', 0)
            env.start()
            model = gp.Model(env=env)

            variables = model.addMVar(shape=len(c), vtype=GRB.CONTINUOUS)
            model.setObjective(c @ variables, GRB.MINIMIZE)
            constraints = model.addConstr(A @ variables == b)
            model.params.Presolve = 0
            if method == 'simplex':
                model.params.Method = 0
            elif method == 'ipm':
                model.params.Method = 2
                model.params.Crossover = 0
#                 model.params.BarConvTol = 1e-4
            else:
                raise Exception(f"Unrecognized LP method '{method}'")
            model.update()

            if basis_start is not None:
                primal_basis, dual_basis = basis_start
                variables.VBasis = primal_basis.cpu().numpy()
                constraints.CBasis = dual_basis.cpu().numpy()
            elif point_start is not None:
                primal_start, dual_start = point_start
                variables.PStart = primal_start.cpu().numpy()
                constraints.DStart = dual_start.cpu().numpy()

            model.optimize()

            optimal_value = torch.FloatTensor([model.objVal]).to(device).squeeze()
            primal_solution = torch.FloatTensor(variables.X).to(device)
            dual_solution = torch.FloatTensor(constraints.Pi).to(device)
            reduced_costs = torch.FloatTensor(variables.RC).to(device)
            if method == 'simplex':
                primal_basis = torch.LongTensor(variables.VBasis).to(device)
                dual_basis = torch.LongTensor(constraints.CBasis).to(device)
            else:
                primal_basis = torch.LongTensor([]).to(device)
                dual_basis = torch.LongTensor([]).to(device)
        
        ctx.save_for_backward(primal_solution, dual_solution)
        ctx.mark_non_differentiable(primal_solution, dual_solution, reduced_costs, primal_basis, dual_basis)
        return optimal_value, primal_solution, dual_solution, reduced_costs, primal_basis, dual_basis
    
    @staticmethod
    def backward(ctx, gradient, *_):
        primal_solution, dual_solution = ctx.saved_tensors
        
        gradient_A = -dual_solution.view(-1, 1) @ primal_solution.view(1, -1) * gradient
        gradient_b = dual_solution * gradient
        gradient_c = primal_solution * gradient
        
        return gradient_A, gradient_b, gradient_c, None, None

def solve_lp(A, b, c, basis_start=None, point_start=None, verbose=False, method='simplex'):
    lp_solution = LinearProgram.apply(A, b, c, basis_start, point_start, verbose, method)
    optimal_value, primal_solution, dual_solution, reduced_costs, primal_basis, dual_basis = lp_solution

    return optimal_value, primal_solution, (primal_basis, dual_basis)


def solve_ilp(A, b, c, vtypes, verbose=False):
    device = A.device
    A, b, c = A.detach().cpu().numpy(), b.detach().cpu().numpy(), c.detach().cpu().numpy()

    with gp.Env(empty=True) as env:
        if not verbose:
            env.setParam('OutputFlag', 0)
        env.start()
        
        model = gp.Model(env=env)
        variables = model.addMVar(shape=len(c), vtype=vtypes)
        model.setObjective(c @ variables, GRB.MINIMIZE)
        constraints = model.addConstr(A @ variables == b)
        model.update()

        model.optimize()
        optimal_value = torch.FloatTensor([model.objVal]).to(device).squeeze()
        optimal_solution = torch.FloatTensor(variables.X).to(device)
    
    return optimal_value, optimal_solution


def solve_lp_from_path(instance_path, verbose=False):
    with gp.Env(empty=True) as env:
        if not verbose:
            env.setParam('OutputFlag', 0)
        env.start()
        
        model = gp.read(instance_path, env=env)
        variables, constraints  = model.getVars(), model.getConstrs()
        for variable in model.getVars():
            variable.VType = GRB.CONTINUOUS
        
        model.optimize()
        
        optimal_value = torch.FloatTensor([model.objVal]).squeeze()
        primal_solution = torch.FloatTensor([variable.X for variable in variables])
        dual_solution = torch.FloatTensor([constraint.Pi for constraint in constraints])
        primal_basis = torch.LongTensor([variable.VBasis for variable in variables])
        dual_basis = torch.LongTensor([constraint.CBasis for constraint in constraints])
    
    return optimal_value, primal_solution, dual_solution, primal_basis, dual_basis


def solve_ilp_from_path(instance_path, verbose=False):
    with gp.Env(empty=True) as env:
        if not verbose:
            env.setParam('OutputFlag', 0)
        env.start()
        
        model = gp.read(instance_path, env=env)
        variables  = model.getVars()
        
        model.optimize()
        
        optimal_value = torch.FloatTensor([model.objVal]).squeeze()
        optimal_solution = torch.FloatTensor([variable.X for variable in variables])
    
    return optimal_value, optimal_solution


def get_basis(A, primal_basis, dual_basis):
    basis = (primal_basis==0).nonzero().squeeze(-1)
    if (dual_basis==0).any():
        basis_extension = torch.cat([A[constraint_index.item(), :].nonzero()[-1]
                                     for constraint_index in (dual_basis==0).nonzero()])
        basis = torch.cat([basis, basis_extension])
    return basis


def pivot(primal_solution, basis, BA, entering_variable, epsilon=1e-8):
    Bn = BA[:, entering_variable]
    Bb = primal_solution[basis]
    t_max = (Bb[Bn>epsilon]/Bn[Bn>epsilon]).min()

    exiting_variable = (Bb[Bn>epsilon]/Bn[Bn>epsilon]).argmin()
    exiting_variable = (Bn>epsilon).nonzero().squeeze(-1)[int(exiting_variable)]
    exiting_variable = basis[exiting_variable]

    new_primal_solution = primal_solution.clone()
    new_primal_solution[basis] = Bb-t_max*Bn
    new_primal_solution[entering_variable] = t_max

    return new_primal_solution.detach()
