import numpy as np
import gurobipy as gp
from gurobipy import GRB
import torch


def load_instance(instance_path, device="cpu", add_variable_bounds=False, presolve=True):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()

        model = gp.read(str(instance_path), env=env)
        if presolve:
            model = model.presolve()

        constraints, variables = model.getConstrs(), model.getVars()
        m, n = len(constraints), len(variables)
        raw_A = model.getA().todense()

        A, b = [], []
        for constraint_index, constraint in enumerate(constraints):
            if constraints[0].Sense in {'>', '='}:
                A.append(raw_A[constraint_index, :])
                b.append(constraint.RHS)
            if constraints[0].Sense in {'<', '='}:
                A.append(-raw_A[constraint_index, :])
                b.append(-constraint.RHS)
        A, b = np.stack(A), np.stack(b)

        c, vtypes = [], []
        for variable_index, variable in enumerate(variables):
            c.append(model.ModelSense*variable.Obj)
            vtypes.append(variable.VType)
        c, vtypes = np.stack(c), np.stack(vtypes)

        # Add variable upper bounds if desired
        if add_variable_bounds:
            upper_bounds_A, upper_bounds_b = [], []
            for variable_index, variable in enumerate(variables):
                if variable.UB is not None and variable.UB < np.inf:
                    upper_bound_A = torch.zeros((1, A.shape[1]))
                    upper_bound_A[0, variable_index] = -1
                    upper_bounds_A.append(upper_bound_A)
                    if variable.VType == GRB.BINARY:
                        upper_bounds_b.append(-1)
                    elif variable.VType == GRB.INTEGER:
                        upper_bounds_b.append(-np.floor(variable.UB))
                    elif variable.VType == GRB.CONTINUOUS:
                        upper_bounds_b.append(-variable.UB)
            upper_bounds_A = np.concatenate(upper_bounds_A, axis=0)
            A = np.concatenate([A, upper_bounds_A], axis=0)
            b = np.concatenate([b, np.array(upper_bounds_b)])

        # Convert lower bounds to >= 0
        variable_index, objective_offset = 0, np.zeros(1)
        for variable in variables:
            if variable.LB is None or variable.LB == -np.inf:
                A = np.insert(A, variable_index+1, -A[:, variable_index], axis=1)
                c = np.insert(c, variable_index+1, -c[variable_index])
                vtypes = np.insert(vtypes, variable_index+1, vtypes[variable_index])
                variable_index += 1
            elif variable.LB != 0:
                b -= A[:, variable_index]*variables[variable_index].LB
                objective_offset += c[variable_index]*variables[variable_index].LB
            variable_index += 1

        return torch.FloatTensor(A).to(device), torch.FloatTensor(b).to(device), \
            torch.FloatTensor(c).to(device), vtypes, objective_offset


def solve_lp(A, b, c, basis_start=None, point_start=None, verbose=False, method='simplex'):
    device, dtype = A.device, A.dtype
    A, b, c = A.detach().cpu().numpy(), b.detach().cpu().numpy(), c.detach().cpu().numpy()

    with gp.Env(empty=True) as env:
        if not verbose:
            env.setParam('OutputFlag', 0)
        env.start()
        model = gp.Model(env=env)

        variables = model.addMVar(shape=len(c), vtype=GRB.CONTINUOUS)
        model.setObjective(c @ variables, GRB.MINIMIZE)
        constraints = model.addConstr(A @ variables >= b)

        model.params.Presolve = 0
        if method == 'simplex':
            model.params.Method = 0
        elif method == 'ipm':
            model.params.Method = 2
            model.params.Crossover = 0
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

        if model.Status == GRB.OPTIMAL:
            optimal_value = model.objVal
            primal_solution = torch.Tensor(variables.X).to(device=device, dtype=dtype)
            dual_solution = torch.Tensor(constraints.Pi).to(device=device, dtype=dtype)
            reduced_costs = torch.Tensor(variables.RC).to(device=device, dtype=dtype)
            if method == 'simplex':
                primal_basis = torch.LongTensor(variables.VBasis).to(device)
                dual_basis = torch.LongTensor(constraints.CBasis).to(device)
            else:
                primal_basis = torch.LongTensor([]).to(device)
                dual_basis = torch.LongTensor([]).to(device)

            return optimal_value, primal_solution, (primal_basis, dual_basis)
        else:
            return None


def solve_ilp(A, b, c, vtypes, verbose=False):
    device, dtype = A.device, A.dtype
    A, b, c = A.detach().cpu().numpy(), b.detach().cpu().numpy(), c.detach().cpu().numpy()

    with gp.Env(empty=True) as env:
        if not verbose:
            env.setParam('OutputFlag', 0)
        env.start()
        
        model = gp.Model(env=env)
        variables = model.addMVar(shape=len(c), vtype=vtypes.tolist())
        model.setObjective(c @ variables, GRB.MINIMIZE)
        constraints = model.addConstr(A @ variables >= b)
        model.update()

        model.optimize()
        if model.Status == GRB.OPTIMAL:
            optimal_value = model.objVal
            optimal_solution = torch.Tensor(variables.X).to(device=device, dtype=dtype)
            return optimal_value, optimal_solution
        else:
            return None

        
def get_joint_basis(A, primal_basis, dual_basis):
    basis = torch.cat([(primal_basis==0).nonzero().squeeze(-1),
                   A.shape[1] + (dual_basis==0).nonzero().squeeze(-1)])
    return basis
