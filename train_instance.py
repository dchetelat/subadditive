import torch
import argparse
from inspect import signature

from lp import *
from layers import *
from gomory import *
from utilities import *
from train_utilities import *

SEED = 0
logger = configure_logging()


class DualFunction(torch.nn.Module):
    def __init__(self, input_size, nb_layers, nonlinear=False):
        super().__init__()
        self.inner_layers = SequentialSubadditive(
            *[Cat(GomoryLayer(input_size + 32*layer, 32, nonlinear)) 
             for layer in range(nb_layers)]
        )
        self.final_layer = LinearLayer(input_size+32*nb_layers, 1)
    
    def forward(self, bias):
        hidden = self.inner_layers(bias)
        return self.final_layer(hidden).squeeze(0)


def train(instance_path, nb_layers=1, gomory_init=False, nonlinear=False, learning_rate=1e-3, target_noise=1e-4, seed=0, gpu=0, nb_steps=10000):
    """
    Train a subadditive neural network to solve the subadditive dual of an instance.
    
    Parameters
    ----------
    instance_path: str or Path
        Path to the instance.
    nb_layers: int
        Number of layers in the subadditive neural network.
    gomory_init: bool
        Should the neural network be initialized from the classical Gomory values?
    nonlinear: bool
        Should nonlinear Gomory cuts be used?
    learning_rate: float
        The learning rate of the optimization.
    target_noise: float
        How much noise should be added to the target in the algorithm?
    seed: int
        Seed to use.
    nb_steps: int
        For how many gradient steps to run the algorithm.
    gpu: int
        Which gpu to use? (cpu=-1)
    """
    short_path_name = Path(instance_path).parent.name + '/' + Path(instance_path).name
    device = f'cuda:{gpu}' if gpu>=0 else 'cpu'
    np.random.seed(seed), torch.manual_seed(seed)
    
    A, b, c, vtypes, lp_value, lp_solution, ilp_value, \
        ilp_solution, gomory_values = get_instance(instance_path, device=device)
    dual_function = DualFunction(len(b), nb_layers, nonlinear).to(device)
    if gomory_init:
        gomory_initialization_(dual_function, A, b, c, vtypes)    
    optimizer = torch.optim.Adam(dual_function.parameters(), lr=learning_rate)

    target, lower_bound = lp_solution, lp_value
    lower_bound = gomory_values[nb_layers] if gomory_init else lp_value 
    lower_bounds, is_step_lp, basis_start = [lower_bound], [True], None
    for step in range(nb_steps):
        extended_A, extended_b, c, vtypes = add_cuts_to_ilp(dual_function.inner_layers, A, b, c, vtypes)
        gap = (extended_A@target - extended_b)[A.shape[0]:].min().item()

        if gap < 1e-5:
            good_rows = torch.max(extended_A.abs().max(-1).values, extended_b.abs()) > 1e-6
            lower_bound, target, basis_start = solve_lp(extended_A[good_rows, :], extended_b[good_rows], c, 
                                                        basis_start=basis_start)
            is_step_lp.append(True)
        else:
            is_step_lp.append(False)
        lower_bounds.append(lower_bound)

        noisy_target = (target + target_noise*torch.randn_like(target)).relu()
        loss = extended_A@noisy_target - extended_b

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        if step % 100 == 0:
            logger.info(f"[{'{i: >{n}}'.format(i=step, n=len(str(nb_steps)))}/{nb_steps}]"
                        f"[{short_path_name: <16}, {nb_layers}L"
                        f", {'G' if gomory_init else 'R'}, {'NL' if nonlinear else 'L': <2}]"
                        f"   lp {lp_value: >6.2f}, nn {lower_bound: >6.2f}"
                        f", gomory {gomory_values[nb_layers]: >6.2f}, optimal {ilp_value: >6.2f}"
                        f"   (gap {gap: >6.2f}, nb lps {np.sum(is_step_lp)})")
    
    return np.array(lower_bounds), np.array(is_step_lp)


if __name__ == "__main__":
    train_parameters = signature(train).parameters
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--instance_path',
        help='Path to instance',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-l', '--nb_layers',
        help='Number of layers',
        type=int,
        default=train_parameters['nb_layers'].default,
    )
    parser.add_argument(
        '-gi', '--gomory_init',
        help='Should the layers be Gomory initialized?',
        type=bool,
        default=train_parameters['gomory_init'].default,
    )
    parser.add_argument(
        '-nl', '--nonlinear',
        help='Should nonlinear cuts be used?',
        type=bool,
        default=train_parameters['gomory_init'].default,
    )
    parser.add_argument(
        '-lr', '--learning_rate',
        help='Learning rate',
        type=float,
        default=train_parameters['learning_rate'].default,
    )
    parser.add_argument(
        '-tn', '--target_noise',
        help='Target noise',
        type=float,
        default=train_parameters['target_noise'].default,
    )
    parser.add_argument(
        '-s', '--seed',
        help='Seed',
        type=int,
        default=train_parameters['seed'].default,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='GPU (-1 for CPU)',
        type=int,
        default=train_parameters['gpu'].default,
    )
    parser.add_argument(
        '-ns', '--nb_steps',
        help='Number of gradient steps to run the training',
        type=int,
        default=train_parameters['nb_steps'].default,
    )
    args = parser.parse_args()
    
    train(**vars(args))
