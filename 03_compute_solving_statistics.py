import pickle
import argparse
import torch
torch.set_default_dtype(torch.float64)
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from gomory import *
from lp import *
from train_utilities import *
from train_instance import CutFunction

logger = configure_logging()


def full_gomory(A, b, c, vtypes, nb_rounds):
    lp_values, values, times = [], [], []
    for _ in range(nb_rounds):
        M, v, gomory_bound = compute_gomory_weights(A, b, c, vtypes)

        layer = Cat(GomoryLayer(len(b), len(v)))
        layer.layer.M.data = M
        layer.layer.v.data = v

        A, b, c, vtypes = add_cuts_to_ilp(layer, A, b, c, vtypes)
    
    return A, b, c, vtypes


def ilp_solving_time(A, b, c, vtypes, verbose=False):
    A, b, c = A.detach().cpu().numpy(), b.detach().cpu().numpy(), c.detach().cpu().numpy()

    with gp.Env(params={'OutputFlag': verbose}) as env:
        env.start()

        model = gp.Model(env=env)
        variables = model.addMVar(shape=len(c), vtype=vtypes.tolist())
        model.setObjective(c @ variables, GRB.MINIMIZE)
        constraints = model.addConstr(A @ variables >= b)
        model.update()
        
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            return model.objVal, model.Runtime
        else:
            return None, None

    
def compute_solving_statistics(results_file_path):
    logger.info(f"Solving {str(results_file_path)}")
    
    with results_file_path.open("rb") as file:
        results = pickle.load(file)
    
    cuts_A, cuts_b, cuts_c, vtypes = results['final_problem']
    nb_cuts = results['size']*results['nb_layers']

    A, b, c = cuts_A[:-nb_cuts], cuts_b[:-nb_cuts], cuts_c

    cut_function = CutFunction(len(b), results['nb_layers'], results['nonlinear'], results['size'])
    gomory_initialization_(cut_function, A, b, c, vtypes)
    gomory_A, gomory_b, gomory_c, vtypes = add_cuts_to_ilp(cut_function.inner_layers, A, b, c, vtypes)

    full_gomory_A, full_gomory_b, full_gomory_c, vtypes = full_gomory(A, b, c, vtypes, results['nb_layers'])

    statistics = {}
    
    statistics['vanilla_lp_value'], *_ = solve_lp(A, b, c)
    statistics['cuts_lp_value'], *_ = solve_lp(cuts_A, cuts_b, cuts_c)
    statistics['gomory_lp_value'], *_ = solve_lp(gomory_A, gomory_b, gomory_c)
    statistics['full_gomory_lp_value'], *_ = solve_lp(full_gomory_A, full_gomory_b, full_gomory_c)
    
    statistics['vanilla_value'], statistics['vanilla_time'] = \
        ilp_solving_time(A, b, c, vtypes)
    statistics['cuts_value'], statistics['cuts_time'] = \
        ilp_solving_time(cuts_A, cuts_b, cuts_c, vtypes)
    statistics['gomory_value'], statistics['gomory_time'] = \
        ilp_solving_time(gomory_A, gomory_b, gomory_c, vtypes)
    statistics['full_gomory_value'], statistics['full_gomory_time'] = \
        ilp_solving_time(full_gomory_A, full_gomory_b, full_gomory_c, vtypes)
    
    results['solving_statistics'] = statistics
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problems',
        help='MILP families to process.',
        nargs='+',
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=12,
    )
    args = parser.parse_args()

    NB_WORKERS = args.njobs

    for problem in args.problems:
        results_folder = Path(f"results/{problem}")
        logger.info(f"Solving {problem}")
        with ThreadPoolExecutor(max_workers=NB_WORKERS, thread_name_prefix='SolverThread') as executor:
            tasks = {}
            for results_file_path in results_folder.glob("*.pkl"):
                future = executor.submit(compute_solving_statistics, results_file_path)
                tasks[future] = results_file_path


            for future in as_completed(tasks):
                results_file_path = tasks[future]
                try:
                    results = future.result()
                    logger.info(f"Updating {str(results_file_path)}")
                    with results_file_path.open("wb") as file:
                        pickle.dump(results, file)

                except Exception as e:
                    logger.error(f"Solving on {str(results_file_path)}"
                                 f" yielded exception", exc_info=e)
        logging.info(f"Done")
