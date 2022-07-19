import argparse
import logging
import concurrent.futures
from pathlib import Path
from itertools import product

from train_instance import train
from train_utilities import *

logger = configure_logging()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--problem',
        help='MILP instance type to process.',
        nargs='*',
        default=[],
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
        help='Which GPU to use (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    NB_WORKERS = args.njobs
    NB_INSTANCES = 100

    if args.problem:
        instance_families = args.problem
    else:
        instance_families = ['setcover', 'cauctions', 'indset', 'facilities', '2-matching']

    for problem in instance_families:
        add_variable_bounds = False
        if problem == 'setcover':
            learning_rate = 5e-4
            target_noise = 1e-4
        elif problem == 'cauctions':
            learning_rate = 5e-4
            target_noise = 1e-4
        elif problem == 'indset':
            learning_rate = 5e-4
            target_noise = 1e-4
        elif problem == 'facilities':
            learning_rate = 5e-4
            target_noise = 1e-4
        elif problem == '2-matching':
            learning_rate = 5e-4
            target_noise = 1e-4
            add_variable_bounds = True
        elif problem == 'small-miplib3':
            learning_rate = 1e-4
            target_noise = 1e-4

        instance_folder = Path(f"instances/{problem}")
        results_folder = Path(f"results/{problem}")
        results_folder.mkdir(parents=True, exist_ok=True)
        instance_paths = sorted(instance_folder.glob("*.lp"), key=path_ordering)[:NB_INSTANCES]

        logger.info(f"Solving {problem}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=NB_WORKERS, thread_name_prefix='SolverThread') as executor:
            futures = {}    
            for config in product(instance_paths, [1, 2], [False, True], [False, True]):
                future = executor.submit(train, *config, learning_rate, target_noise, args.seed, args.gpu, add_variable_bounds)
                futures[future] = config

            for future in concurrent.futures.as_completed(futures):
                instance_path, nb_layers, gomory_init, nonlinear = futures[future]
                try:
                    lower_bounds, is_step_lp = future.result()
                    results = {"instance_path": instance_path, 
                               "nb_layers": nb_layers,
                               "gomory_init": gomory_init,
                               "nonlinear": nonlinear,
                               "learning_rate": learning_rate,
                               "target_noise": target_noise,
                               "seed": args.seed,
                               "lower_bounds": lower_bounds,
                               "is_step_lp": is_step_lp}

                    results_file = f"{instance_path.stem}_{nb_layers}_{'g' if gomory_init else 'r'}_{'n' if nonlinear else 'l'}.pkl"
                    with (results_folder/results_file).open("wb") as file:
                        pickle.dump(results, file)
                
                except Exception as e:
                    logger.error(f"Solving on {instance_path} with nb_layers={nb_layers}, gomory_init={gomory_init}"
                          f", nonlinear={nonlinear} yielded exception", exc_info=e)

            logging.info(f"Done")