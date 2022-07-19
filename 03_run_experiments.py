import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from train_instance import train
from train_utilities import *

logger = configure_logging()

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
            config['add_variable_bounds'] = False
        elif problem == 'cauctions':
            config['learning_rate'] = 5e-4
            config['add_variable_bounds'] = False
        elif problem == 'indset':
            config['learning_rate'] = 5e-4
            config['add_variable_bounds'] = False
        elif problem == 'facilities':
            config['learning_rate'] = 1e-4
            config['add_variable_bounds'] = False
        elif problem == '2-matching':
            config['learning_rate'] = 1e-4
            config['add_variable_bounds'] = True
        elif problem == 'small-miplib3':
            config['learning_rate'] = 1e-4
            config['add_variable_bounds'] = True

        instance_folder = Path(f"instances/{problem}")
        results_folder = Path(f"results/{problem}")
        results_folder.mkdir(parents=True, exist_ok=True)
        instance_paths = sorted(instance_folder.glob("*.lp"), key=path_ordering)[:NB_INSTANCES]

        logger.info(f"Solving {problem}")
        with ThreadPoolExecutor(max_workers=NB_WORKERS, thread_name_prefix='SolverThread') as executor:
            configs = {}
            for parameters in dict_product(instance_path=instance_paths, nb_layers=[1, 2], 
                                           gomory_init=[False, True], nonlinear=[False, True]):
                config.update(parameters)
                future = executor.submit(train, **config)
                configs[future] = config

            for future in as_completed(configs):
                config = configs[future]
                try:
                    lower_bounds, is_step_lp = future.result()
                    results = {**config,
                               "lower_bounds": lower_bounds,
                               "is_step_lp": is_step_lp}

                    results_file = f"{config['instance_path'].stem}" + \
                                   f"_{config['nb_layers']}" + \
                                   f"_{'g' if config['gomory_init'] else 'r'}" + \
                                   f"_{'n' if config['nonlinear'] else 'l'}.pkl"
                    with (results_folder/results_file).open("wb") as file:
                        pickle.dump(results, file)
                
                except Exception as e:
                    logger.error(f"Solving on {config['instance_path']}"
                                 f" with nb_layers={config['nb_layers']}"
                                 f", gomory_init={config['gomory_init']}"
                                 f", nonlinear={config['nonlinear']}"
                                 f" yielded exception", exc_info=e)

            logging.info(f"Done")