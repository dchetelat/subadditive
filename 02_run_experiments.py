import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from train_instance import train
from train_utilities import *

logger = configure_logging("train_log.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problems',
        help='MILP families to process.',
        nargs='+',
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

    config = {'seed': args.seed, 'gpu': args.gpu, 'target_noise': 1e-4}
    for problem in args.problems:
        if problem == 'setcover':
            config['learning_rate'] = 5e-4
            config['size'] = 32
            config['add_variable_bounds'] = False
            config['nb_steps'] = 10000
        elif problem == 'cauctions':
            config['learning_rate'] = 5e-4
            config['size'] = 32
            config['add_variable_bounds'] = False
            config['nb_steps'] = 10000
        elif problem == 'indset':
            config['learning_rate'] = 5e-4
            config['size'] = 32
            config['add_variable_bounds'] = False
            config['nb_steps'] = 10000
        elif problem == 'facilities':
            config['learning_rate'] = 1e-4
            config['size'] = 32
            config['add_variable_bounds'] = False
            config['nb_steps'] = 10000
        elif problem == '2-matching':
            config['learning_rate'] = 1e-4
            config['size'] = 1024
            config['add_variable_bounds'] = True
            config['nb_steps'] = 10000
        elif problem == 'small-miplib3':
            config['learning_rate'] = 1e-4
            config['size'] = 1024
            config['add_variable_bounds'] = True
            config['nb_steps'] = 50000

        instance_folder = Path(f"instances/{problem}")
        results_folder = Path(f"results/{problem}-renormalized") # DEBUG
        results_folder.mkdir(parents=True, exist_ok=True)
        instance_paths = list(instance_folder.glob("*.lp"))+list(instance_folder.glob("*.mps.gz"))
        instance_paths = sorted(instance_paths, key=path_ordering)[:NB_INSTANCES]

        logger.info(f"Solving {problem} [{len(instance_paths)} instances]")
        with ThreadPoolExecutor(max_workers=NB_WORKERS, thread_name_prefix='SolverThread') as executor:
            train_configs = {}
            for parameters in dict_product(instance_path=instance_paths, nb_layers=[1, 2], 
                                           gomory_init=[False, True], nonlinear=[False, True]):
                
                train_config = {**config, **parameters}
                
                results_file = f"{train_config['instance_path'].stem}" + \
                                   f"_{train_config['nb_layers']}" + \
                                   f"_{'g' if train_config['gomory_init'] else 'r'}" + \
                                   f"_{'n' if train_config['nonlinear'] else 'l'}.pkl"
                if not (results_folder/results_file).is_file():
                    future = executor.submit(train, **train_config)
                    train_configs[future] = train_config

            for future in as_completed(train_configs):
                train_config = train_configs[future]
                try:
                    lower_bounds, is_step_lp, nb_targets, final_problem, ilp_value, solving_time = future.result()
                    results = {**train_config,
                               "lower_bounds": lower_bounds,
                               "is_step_lp": is_step_lp,
                               "nb_targets": nb_targets,
                               "final_problem": final_problem,
                               "ilp_value": ilp_value,
                               "solving_time": solving_time}

                    results_file = f"{train_config['instance_path'].stem}" + \
                                   f"_{train_config['nb_layers']}" + \
                                   f"_{'g' if train_config['gomory_init'] else 'r'}" + \
                                   f"_{'n' if train_config['nonlinear'] else 'l'}.pkl"
                    logger.info(f"Saving {results_file}")
                    with (results_folder/results_file).open("wb") as file:
                        pickle.dump(results, file)
                
                except Exception as e:
                    logger.error(f"Solving on {train_config['instance_path']}"
                                 f" with nb_layers={train_config['nb_layers']}"
                                 f", gomory_init={train_config['gomory_init']}"
                                 f", nonlinear={train_config['nonlinear']}"
                                 f" yielded exception", exc_info=e)

            logging.info(f"Done")