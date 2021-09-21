import logging
import concurrent.futures
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from itertools import product
from utilities import *
from lp import *
from layers import *
from gomory import *


class SparseDualFunction(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.inner_layers = SequentialSubadditive(
            SparseCat(SparseGomoryLayer(input_size, 32)),
            SparseCat(SparseGomoryLayer(input_size+32, 32)),
        )
        self.final_layer = LinearLayer(input_size+32+32, 1)
    
    def forward(self, bias):
        hidden = self.inner_layers(bias)
        return self.final_layer(hidden).squeeze(0)


def solve_instance(instance_path, save_folder, seed, gomory_initialization):
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)

        A, b, c, vtypes = load_instance(str(instance_path), device="cpu", add_variable_bounds=False, presolve=True)
        optimal_value, optimal_solution = solve_ilp(A, b, c, vtypes)
        
        if OBJECTIVE_NOISE > 0:
            c = perturb_objective(A, b, c, vtypes, optimal_solution)
        
        gomory_bounds = compute_gomory_bounds(A, b, c, vtypes, nb_rounds=2)
        lp_value = gomory_bounds[0]
        info = {"instance_path": instance_path.name, "seed": seed,
                "problem_shape": np.array(A.shape), "gomory_initialization": gomory_initialization,
                "optimal_value": optimal_value.item(),  "lp_optimal_value": lp_value, 
                "gomory": gomory_bounds[2]}

        if lp_value == optimal_value:
            logging.warning(f"Instance {instance_path} was solved in presolving (LP=ILP value), skipping")
        else:
            nn_best_bounds = train_subadditive(A, b, c, vtypes, info, gomory_initialization)
            info["nn_best_bounds"] = np.array(nn_best_bounds)

            save_folder.mkdir(parents=True, exist_ok=True)
            save_file = f"{instance_path.stem}_{seed}_{'g' if gomory_initialization else 'r'}.pkl"
            with (save_folder/save_file).open("wb") as file:
                pickle.dump(info, file)

    except Exception as e:
        logging.exception(f"Exception caught while solving {instance_path.name}")
        raise e


def train_subadditive(A, b, c, vtypes, info, gomory_initialization):
    device = DEVICE
    dual_function = SparseDualFunction(input_size=len(b)).to(device)
    if gomory_initialization:
        gomory_initialization_(dual_function, A, b, c, vtypes)
    optimizer = torch.optim.Adam(dual_function.parameters(), lr=LEARNING_RATE)
    
    integral_vars = np.isin(vtypes, [GRB.BINARY, GRB.INTEGER])
    continuous_vars = np.isin(vtypes, [GRB.CONTINUOUS])
    integral_A = A[:, integral_vars].to_sparse().to(device)
    continuous_A = A[:, continuous_vars].to_sparse().to(device)
    b = b.to(device)
    integral_c = c[integral_vars].to(device)
    continuous_c = c[continuous_vars].to(device)
    
    best_bound, best_bounds = -np.inf, []
    target, basis_start = None, None

    for step in range(NB_ITERATIONS):
        loss_integral_A, loss_continuous_A, loss_b, loss_integral_c, loss_continuous_c, loss_vtypes = \
            integral_A, continuous_A, b, integral_c, continuous_c, vtypes
        for layer in dual_function.inner_layers:
            loss_integral_A, loss_continuous_A, loss_b, loss_integral_c, loss_continuous_c, loss_vtypes = \
                get_sparse_extended_lp(loss_integral_A, loss_continuous_A, loss_b, loss_integral_c, loss_continuous_c, loss_vtypes, layer)
        loss_A = sparse_cat(loss_integral_A.cpu(), loss_continuous_A.cpu(), 1).to_dense()
        loss_b = loss_b.cpu()
        loss_c = torch.cat([loss_integral_c.cpu(), loss_continuous_c.cpu()])

        cut_gap = -np.inf if target is None else get_gaps(A.shape, loss_A, loss_b, target, dual_function.inner_layers).min()
        if cut_gap < 0:
            lower_bound, target, basis_start = solve_lp(loss_A, loss_b, loss_c, basis_start=basis_start)
            best_bound = max(lower_bound.item(), best_bound)
        best_bounds.append(best_bound)

        noisy_target = target + TARGET_NOISE*torch.randn_like(target)
        loss = loss_A[A.shape[0]:]@noisy_target - loss_b[A.shape[0]:].cpu()
        loss = -torch.linalg.solve(loss_A[A.shape[0]:, A.shape[1]:], loss.unsqueeze(-1)).squeeze(-1)

        if step % (NB_ITERATIONS//40) == 0:
            gap = (info['optimal_value']-best_bounds[-1])/(info['optimal_value'] - info['lp_optimal_value'])
            gomory_gap = (info['optimal_value']-info['gomory'])/(info['optimal_value'] - info['lp_optimal_value'])
            logging.info(f"Iteration {step: >5}/{NB_ITERATIONS: <5} [{info['instance_path']: <15} / seed {info['seed']} / Gomory {str(gomory_initialization): >5}]"
                f" | NN gap {100*gap: >6.2f}%, Gomory gap  {100*gomory_gap: >6.2f}% [gain {100*(gomory_gap-gap): >6.2f}%]"
                f", cut gap {cut_gap: >10.6f}")
        
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    return best_bounds


def get_gaps(A_shape, loss_A, loss_b, target, layers):
    gaps = []
    row_index, column_index = A_shape
    for layer in layers:
        nb_cuts = layer.layer.out_size
        gap = loss_A[row_index:row_index+nb_cuts, :column_index]@target[:column_index] - loss_b[row_index:row_index+nb_cuts]
        gaps.append(gap)

        row_index += nb_cuts
        column_index += nb_cuts
    return torch.cat(gaps)


def perturb_objective(A, b, c, vtypes, optimal_solution):
    while True:
        c_prime = c + OBJECTIVE_NOISE*torch.randn_like(c)
        _, solution = solve_ilp(A, b, c_prime, vtypes)
        if solution@c == optimal_solution@c:
            break
    return c_prime


# --------------------------------------------------------------------------------------------------------

# Main code

NB_WORKERS = 14
NB_INSTANCES = 100
NB_ITERATIONS = 10000
DEVICE = "cuda"

# instance_set = "indset"
# LEARNING_RATE = 1e-4
# TARGET_NOISE = 1e-4
# OBJECTIVE_NOISE = 1e-3

instance_set = "facilities"
LEARNING_RATE = 1e-4
TARGET_NOISE = 1e-4
OBJECTIVE_NOISE = 1e-3

# Two-layer learning rates:
# Setcover: 1e-3
# Cauctions: 1e-3
# Indset: 1e-4
# Facilities: 1e-4

# One-layer learning rates:
# Setcover: 1e-3
# Cauctions: 1e-3
# Indset: 1e-4
# Facilities: 5e-4


instance_folder = Path(f"data/instances/{instance_set}/train")
save_folder = Path(f"data/andrea-experiment-mixed/{instance_set}")

logging.basicConfig(
    format='[%(asctime)s %(levelname)-7s]  %(threadName)-23s  |  %(message)s',
    level=logging.INFO, datefmt='%H:%M:%S')

save_folder.mkdir(exist_ok=True, parents=True)
instances = list(instance_folder.glob("*.lp"))[:NB_INSTANCES]

with concurrent.futures.ThreadPoolExecutor(max_workers=NB_WORKERS) as executor:
    logging.info(f"Solving {instance_folder}")
    futures = [executor.submit(solve_instance, instance_path, save_folder, seed, gomory_init)
                               for instance_path, seed, gomory_init 
                               in product(instances, [0], [True, False])]
    concurrent.futures.wait(futures)
    logging.info(f"Done")
torch.Size([9831, 19931])