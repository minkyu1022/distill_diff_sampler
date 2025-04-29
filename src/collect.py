import os
import wandb
import argparse

from utils import *
from teachers import *
from energies import *
from plot_utils import *
from metrics.evaluations import *
from metrics.gflownet_losses import *

parser = argparse.ArgumentParser()

# System config
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--date', type=str, default='test')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--method', type=str, default='ours')
parser.add_argument('--project', type=str, default='aldp')

# Dataset config
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--teacher', type=str, default='md', choices=('md', 'mala'))
parser.add_argument('--energy', type=str, default='aldp', choices=('aldp', 'lj13', 'lj55'))

## MD config
parser.add_argument("--gamma", default=1.0, type=float)
parser.add_argument('--n_steps', type=int, default=200000)
parser.add_argument("--timestep", default=5e-4, type=float)
parser.add_argument("--temperature", default=600, type=float)
parser.add_argument('--teacher_batch_size', type=int, default=300)

# MALA config
parser.add_argument('--burn_in', type=int, default=15000)
parser.add_argument('--rnd_weight', type=float, default=0)
parser.add_argument('--prior_std', type=float, default=1.75)
parser.add_argument('--ld_step', type=float, default=0.00001)
parser.add_argument('--max_iter_ls', type=int, default=20000)
parser.add_argument('--ld_schedule', action='store_true', default=False)
parser.add_argument('--target_acceptance_rate', type=float, default=0.574)

args = parser.parse_args()

set_seed(args.seed)

def get_energy():
    if args.energy == 'aldp':
        energy = ALDP(args)
    elif args.energy == 'lj13':
        energy = LJ13(args)
    elif args.energy == 'lj55':
        energy = LJ55(args)
    return energy

def get_teacher():
    if args.teacher == 'md':
        teacher = MD(args, energy)
    elif args.teacher == 'mala':
        teacher = MALA(args, energy)
    return teacher


if __name__ == '__main__':    
    name = f'data/{args.energy}/{args.teacher}'
    if not os.path.exists(name):
        for subdir in ['positions', 'rewards']:
            os.makedirs(f'{name}/{subdir}')

    wandb.init(project=args.project, config=args.__dict__)
    wandb.run.log_code(".")

    energy = get_energy()
    teacher = get_teacher()
    
    global_epochs = 0

    if args.teacher=='mala':
        prior = Gaussian(args.device, energy.data_ndim, std=args.prior_std)
        initial_positions = prior.sample(args.teacher_batch_size).to(args.device)
    elif args.teacher=='md':
        initial_positions = energy.initial_position
    samples, rewards = teacher.sample(initial_positions)

    np.save(f'{name}/positions.npy', samples.detach().cpu().numpy())
    np.save(f'{name}/rewards.npy', rewards.detach().cpu().numpy())