import re
import sys
import logging
import pickle
from pathlib import Path
from itertools import product

from lp import *
from gomory import *


def get_instance(instance_path, add_variable_bounds=False, presolve=True, force_reload=False, device="cpu"):
    instance_path = Path(instance_path)
    
    # Obtain solution info
    solution_path = instance_path.parent/"solutions"/(instance_path.stem+".pkl")
    if solution_path.exists() and not force_reload:
        # Load solutions
        with open(solution_path, "rb") as solution_file:
            solutions = pickle.load(solution_file)
        solutions = tuple(t.to(device) if torch.is_tensor(t) else t for t in solutions)
    else:
        # Compute solutions
        A, b, c, vtypes, _ = load_instance(instance_path, device=device, 
                                           add_variable_bounds=add_variable_bounds, presolve=presolve)
        lp_value, lp_solution, _ = solve_lp(A, b, c)
        ilp_value, ilp_solution = solve_ilp(A, b, c, vtypes)
        gomory_values = compute_gomory_bounds(A, b, c, vtypes, nb_rounds=2)
        solutions = A, b, c, vtypes, lp_value, lp_solution, ilp_value, ilp_solution, gomory_values
        
        # Save solutions
        solutions_on_cpu = tuple(t.to("cpu") if torch.is_tensor(t) else t for t in solutions)
        solution_path.parent.mkdir(exist_ok=True)
        with open(solution_path, "wb") as solution_file:
            pickle.dump(solutions_on_cpu, solution_file)
    
    return solutions


def configure_logging(output_file=None):
    logger = logging.getLogger("subadditive")
    
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(datefmt='%H:%M:%S',
#             fmt='[%(asctime)s]  %(threadName)-12s  %(message)s'
            fmt='[%(asctime)s] %(message)s')

        if output_file:
            file_handler = logging.FileHandler(output_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def path_ordering(path):
    groups = re.split("[/_.]", str(path))
    groups = [int(group) if group.isnumeric() else group for group in groups]
    return groups

def dict_product(**iterable):
    for items in product(*iterable.values()):
        yield dict(zip(iterable.keys(), items))
