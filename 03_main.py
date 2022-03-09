import argparse
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


class DualFunction(torch.nn.Module):
    def __init__(self, input_size, nb_layers):
        super().__init__()
        self.inner_layers = SequentialSubadditive(
            *[Cat(GomoryLayer(input_size + 32*layer, 32)) 
             for layer in range(nb_layers)]
        )
        self.final_layer = LinearLayer(input_size+32*nb_layers, 1)
    
    def forward(self, bias):
        hidden = self.inner_layers(bias)
        return self.final_layer(hidden).squeeze(0)


def solve_instance(instance_path, instance_info, save_folder, nb_layers, seed, gomory_initialization):
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)

        A, b, c, vtypes, objective_offset = load_instance(str(instance_path), device="cpu", 
                                                          add_variable_bounds=instance_info["add_variable_bounds"], 
                                                          presolve=instance_info["presolve"])
        optimal_value = instance_info["optimal_value"] - objective_offset
        gomory_bounds = compute_gomory_bounds(A, b, c, vtypes, nb_rounds=nb_layers)
        lp_optimal_value = gomory_bounds[0]
        
        if OBJECTIVE_NOISE > 0:
            _, optimal_solution = solve_ilp(A, b, c, vtypes)
            perturbed_c = perturb_objective(A, b, c, vtypes, optimal_solution)
            perturbed_optimal_value = optimal_solution@perturbed_c
            perturbed_gomory_bounds = compute_gomory_bounds(A, b, perturbed_c, vtypes, nb_rounds=nb_layers)
        else:
            perturbed_optimal_value = optimal_value
            perturbed_gomory_bounds = gomory_bounds
        perturbed_lp_optimal_value = perturbed_gomory_bounds[0]
        
        info = {"instance_path": instance_path.name, "nb_layers": nb_layers, "seed": seed, 
                "problem_shape": np.array(A.shape), "gomory_initialization": gomory_initialization,
                "optimal_value": optimal_value.item(), 
                "lp_optimal_value": lp_optimal_value, 
                "gomory": gomory_bounds[-1],
                "perturbed_optimal_value": perturbed_optimal_value.item(), 
                "perturbed_lp_optimal_value": perturbed_lp_optimal_value, 
                "perturbed_gomory": perturbed_gomory_bounds[-1]}

        if lp_optimal_value == optimal_value:
            logging.warning(f"Instance {instance_path} was solved in presolving (LP=ILP value), skipping")
        else:
            nn_best_bounds, nn_best_perturbed_bounds = train_subadditive(A, b, c, perturbed_c, vtypes, info, nb_layers, gomory_initialization)
            info["nn_best_bounds"] = np.array(nn_best_bounds)
            info["nn_best_perturbed_bounds"] = np.array(nn_best_perturbed_bounds)

            save_folder.mkdir(parents=True, exist_ok=True)
            save_file = f"{instance_path.stem}_{nb_layers}_{seed}_{'g' if gomory_initialization else 'r'}.pkl"
            with (save_folder/save_file).open("wb") as file:
                pickle.dump(info, file)

    except Exception as e:
        logging.exception(f"Exception caught while solving {instance_path.name}")
        raise e


def train_subadditive(A, b, c, perturbed_c, vtypes, info, nb_layers, gomory_initialization):
    problem_description = get_problem_description(A, b, perturbed_c, vtypes, DEVICE, sparse=USE_SPARSE_TENSORS)
    dual_function = DualFunction(input_size=len(b), nb_layers=nb_layers)
    if gomory_initialization:
        gomory_initialization_(dual_function, A, b, c, vtypes) # or perturbed_c?
    dual_function = dual_function.to(DEVICE)
    optimizer = torch.optim.Adam(dual_function.parameters(), lr=LEARNING_RATE)
    
    best_bound, best_bounds = -np.inf, []
    best_perturbed_bound, best_perturbed_bounds = -np.inf, []
    target, basis_start = None, None
    for step in range(NB_ITERATIONS):
        loss_A, loss_b, loss_c, loss_vtypes = get_extended_problem_description(problem_description, 
                                                                               dual_function, sparse=USE_SPARSE_TENSORS)
        
        cut_gap = -np.inf if target is None else get_gaps(A.shape, loss_A, loss_b, target, dual_function.inner_layers).min()
        if cut_gap < 0:
            lower_perturbed_bound, target, basis_start = solve_lp(loss_A, loss_b, loss_c, basis_start=basis_start)
            lower_bound = target[:A.shape[1]]@c
            best_bound = max(lower_bound.item(), best_bound)
            best_perturbed_bound = max(lower_perturbed_bound.item(), best_perturbed_bound)
        best_bounds.append(best_bound), best_perturbed_bounds.append(best_perturbed_bound)
        
        if all(target[:A.shape[1]] == target[:A.shape[1]].long()):
            logging.info(f"Found optimal solution of problem {info['instance_path']} with {info['nb_layers']} layer(s), early stopping")
            break

        noisy_target = target + TARGET_NOISE*torch.randn_like(target)
        noisy_target = noisy_target.relu()
        loss = loss_A[A.shape[0]:]@noisy_target - loss_b[A.shape[0]:]
        loss = -torch.linalg.solve(loss_A[A.shape[0]:, A.shape[1]:], loss.unsqueeze(-1)).squeeze(-1)

        if step % (NB_ITERATIONS//40) == 0:
            gap = (info['optimal_value']-best_bounds[-1])/(info['optimal_value'] - info['lp_optimal_value'])
            gomory_gap = (info['optimal_value']-info['gomory'])/(info['optimal_value'] - info['lp_optimal_value'])
            logging.info(f"Iteration {step: >5}/{NB_ITERATIONS: <5} [{info['instance_path']: <16}"
                         f" / {nb_layers} layer(s) / seed {info['seed']} / Gomory {str(gomory_initialization): >5}]"
                         f" | NN gap {100*gap: >6.2f}%, Gomory gap {100*gomory_gap: >6.2f}% [gain {100*(gomory_gap-gap): >7.2f}%]"
                         f", cut gap {cut_gap: >10.6f}")
        
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        
    best_bounds += [best_bounds[-1] for _ in range(NB_ITERATIONS-len(best_bounds))]
    best_perturbed_bounds += [best_perturbed_bounds[-1] for _ in range(NB_ITERATIONS-len(best_perturbed_bounds))]

    return best_bounds, best_perturbed_bounds


def get_problem_description(A, b, c, vtypes, device, sparse):
    if sparse:
        integral_vars = np.isin(vtypes, [GRB.BINARY, GRB.INTEGER])
        continuous_vars = np.isin(vtypes, [GRB.CONTINUOUS])
        integral_A = A[:, integral_vars].to_sparse().to(device)
        continuous_A = A[:, continuous_vars].to_sparse().to(device)
        b = b.to(device)
        integral_c = c[integral_vars].to(device)
        continuous_c = c[continuous_vars].to(device)
        return integral_A, continuous_A, b, integral_c, continuous_c, vtypes
    else:
        return A.to(device), b.to(device), c.to(device), vtypes


def get_extended_problem_description(problem_description, dual_function, sparse):
    if sparse:
        for layer in dual_function.inner_layers:
            problem_description = get_sparse_extended_lp(*problem_description, layer)
        loss_integral_A, loss_continuous_A, loss_b, loss_integral_c, loss_continuous_c, loss_vtypes = problem_description
        loss_A = sparse_cat(loss_integral_A.cpu(), loss_continuous_A.cpu(), 1).to_dense()
        loss_b = loss_b.cpu()
        loss_c = torch.cat([loss_integral_c.cpu(), loss_continuous_c.cpu()])
    else:
        for layer in dual_function.inner_layers:
            problem_description = get_extended_lp(*problem_description, layer)
        loss_A, loss_b, loss_c, loss_vtypes = problem_description
    return loss_A, loss_b, loss_c, loss_vtypes


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset', '2-matching', 'small-miplib3'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=12,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--dense',
        help='Should dense tensors be used?',
        default=False,
    )
    args = parser.parse_args()
    
    NB_WORKERS = args.njobs
    NB_INSTANCES = 100
    DEVICE = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    USE_SPARSE_TENSORS = not args.dense
    
    # Two-layer learning rates:
    # Setcover: 1e-3
    # Cauctions: 1e-3
    # Indset: 1e-3
    # Facilities: 1e-3
    # Indset-old: 1e-4
    # Facilities-old: 1e-4

    # One-layer learning rates:
    # Setcover: 1e-3
    # Cauctions: 1e-3
    # Indset: 1e-4
    # Facilities: 5e-4
    
    if args.problem in 'setcover':
        LEARNING_RATE = 1e-3
        TARGET_NOISE = 1e-4
        OBJECTIVE_NOISE = 1e-3
        NB_ITERATIONS = 10000
    elif args.problem == 'cauctions':
        LEARNING_RATE = 1e-3
        TARGET_NOISE = 1e-4
        OBJECTIVE_NOISE = 1e-3
        NB_ITERATIONS = 10000
    elif args.problem == 'indset':
        LEARNING_RATE = 1e-4
        TARGET_NOISE = 1e-4
        OBJECTIVE_NOISE = 1e-3
        NB_ITERATIONS = 10000
    elif args.problem == 'facilities':
        LEARNING_RATE = 5e-4
        TARGET_NOISE = 1e-4
        OBJECTIVE_NOISE = 1e-3
        NB_ITERATIONS = 10000
    elif args.problem == '2-matching':
        LEARNING_RATE = 1e-4
        TARGET_NOISE = 5e-4
        OBJECTIVE_NOISE = 1e-3
        NB_ITERATIONS = 50000
    elif args.problem == 'small-miplib3':
        LEARNING_RATE = 1e-4
        TARGET_NOISE = 1e-4
        OBJECTIVE_NOISE = 0
        NB_ITERATIONS = 100000
    
    
    # ----------------------------------------------------------------------------
    
    instance_folder = Path(f"data/instances/{args.problem}")
    save_folder = Path(f"data/ipco/{args.problem}")

    logging.basicConfig(
        format='[%(asctime)s %(levelname)-7s]  %(threadName)-23s  |  %(message)s',
        level=logging.INFO, datefmt='%H:%M:%S')

    save_folder.mkdir(exist_ok=True, parents=True)
    
    with (instance_folder/"solution_info.pkl").open('rb') as file:
        instance_info = pickle.load(file)
    
    instance_names = list(instance_info)[:NB_INSTANCES]

    if USE_SPARSE_TENSORS:
        GomoryLayer = SparseGomoryLayer
        Cat = SparseCat

    with concurrent.futures.ThreadPoolExecutor(max_workers=NB_WORKERS) as executor:
        logging.info(f"Solving {instance_folder}")
        futures = [executor.submit(solve_instance, instance_folder/instance_name, instance_info[instance_name], 
                                   save_folder, nb_layers, seed, gomory_init)
                                   for instance_name, nb_layers, seed, gomory_init 
                                   in product(instance_names, [1, 2], [args.seed], [True, False])]
        concurrent.futures.wait(futures)
        logging.info(f"Done")
