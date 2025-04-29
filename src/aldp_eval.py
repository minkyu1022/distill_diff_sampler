import os
import wandb
import torch
import argparse

from utils import *
from models import GFN
from energies import *
from plot_utils import *
from buffer import ReplayBuffer
from metrics.evaluations import *
from metrics.gflownet_losses import *

parser = argparse.ArgumentParser()

# System config
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--date', type=str, default='test')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--project', type=str, default='aldp')

# Dataset config
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--t_scale', type=float, default=0.2)
parser.add_argument("--temperature", default=300, type=float)
parser.add_argument('--log_var_range', type=float, default=4.)
parser.add_argument('--energy', type=str, default='aldp', choices=('aldp'))

# Architecture config
parser.add_argument('--architecture', type=str, default="egnn", choices=('pis', 'egnn'))

## Dimensions
parser.add_argument('--s_emb_dim', type=int, default=64)
parser.add_argument('--t_emb_dim', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--harmonics_dim', type=int, default=64)

## Layers
parser.add_argument('--lgv_layers', type=int, default=3)
parser.add_argument('--joint_layers', type=int, default=5)
parser.add_argument('--zero_init', action='store_true', default=False)

## GFN
parser.add_argument('--T', type=int, default=100)
parser.add_argument('--lgv_clip', type=float, default=1e2)
parser.add_argument('--gfn_clip', type=float, default=1e4)
parser.add_argument('--sigma_max', type=float, default=0.05)
parser.add_argument('--sigma_min', type=float, default=0.00001)
parser.add_argument('--pb_scale_range', type=float, default=0.1)
parser.add_argument('--bwd', action='store_true', default=False)
parser.add_argument('--langevin', action='store_true', default=False)
parser.add_argument('--clipping', action='store_true', default=False)
parser.add_argument('--learn_pb', action='store_true', default=False)
parser.add_argument('--partial_energy', action='store_true', default=False)
parser.add_argument('--learned_variance', action='store_true', default=False)
parser.add_argument('--conditional_flow_model', action='store_true', default=False)
parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
parser.add_argument('--scheduler', type=str, default='linear', choices=('linear', 'geometric'))

## Replay buffer
parser.add_argument('--beta', type=float, default=1.)
parser.add_argument('--rank_weight', type=float, default=1e-2)
parser.add_argument('--buffer_size', type=int, default=1200000)
parser.add_argument('--prioritized', type=str, default="rank", choices=('none', 'reward', 'rank'))
parser.add_argument('--sampling', type=str, default="buffer", choices=('sleep_phase', 'energy', 'buffer'))

# Logging config
parser.add_argument('--eval_size', type=int, default=50000)

args = parser.parse_args()

def get_energy():
    if args.energy == 'aldp':
        energy = ALDP(args)
    elif args.energy == 'lj13':
        energy = LJ13(args)
    elif args.energy == 'lj55':
        energy = LJ55(args)
    return energy

def eval(energy, buffer, gfn_model):
    metrics = dict()
    
    init_state = torch.zeros(5000, energy.data_ndim).to(args.device)
    gt_samples = energy.sample(5000).to(energy.device)
    teacher_samples = buffer.sample_pos(5000).to(args.device)
    
    samples, metrics['final_eval/log_Z_IS'], metrics['final_eval/ELBO'], metrics['final_eval/log_Z_learned'] = log_partition_function(init_state, gfn_model, energy.log_reward)

    metrics['final_eval/EUBO'] = EUBO(gt_samples, gfn_model, energy.log_reward)

    energies = energy.energy(samples)
    gt_energies = energy.energy(gt_samples)
    teacher_energies = energy.energy(teacher_samples)
    energy_dict = {
        'Student': energies.detach().cpu().numpy(),
        'GT': gt_energies.detach().cpu().numpy(),
        'Teacher': teacher_energies.detach().cpu().numpy(),
    }
    energy_hist_fig = plot_energy_hist(energy_dict)
    
    metrics.update(get_sample_metrics(samples, gt_samples, True))
    
    sampless = []
    count = 0
    while count < args.eval_size:
        init_state = torch.zeros(5000, energy.data_ndim).to(args.device)
        samples, metrics['final_eval/log_Z_IS'], metrics['final_eval/ELBO'], metrics['final_eval/log_Z_learned'] = log_partition_function(init_state, gfn_model, energy.log_reward)
        samples = align_topologies(samples)
        samples = align_chiral(samples)
        samples = samples.reshape(samples.shape[0], -1)
        sampless.append(samples)
        
        count += samples.shape[0]
        print(f'Sampled {count}/{args.eval_size}={count/args.eval_size:.2f}')
    
    samples = torch.cat(sampless, dim=0)
    samples = samples[:args.eval_size]
    gt_samples = energy.sample(args.eval_size).to(energy.device)
    teacher_samples = buffer.sample_pos(args.eval_size).to(args.device)
    
    phi_psi_fig = plot_phi_psi(samples.reshape(samples.shape[0], -1, 3))
    gt_phi_psi_fig = plot_phi_psi(gt_samples.reshape(gt_samples.shape[0], -1, 3))
    teacher_phi_psi_fig = plot_phi_psi(teacher_samples.reshape(teacher_samples.shape[0], -1, 3))
    aldp_fig = draw_aldps(samples[:3])

    metrics["visualization/energy_hist"] = wandb.Image(fig_to_image(energy_hist_fig))
    metrics["visualization/phi_psi"] = wandb.Image(fig_to_image(phi_psi_fig))
    metrics["visualization/gt_phi_psi"] = wandb.Image(fig_to_image(gt_phi_psi_fig))
    metrics["visualization/teacher_phi_psi"] = wandb.Image(fig_to_image(teacher_phi_psi_fig))
    metrics["visualization/aldp"] = wandb.Image(fig_to_image(aldp_fig))

    np.save(f'{name}/samples.npy', samples.reshape(samples.shape[0], -1, 3).cpu().numpy())
    np.save(f'{name}/energies.npy', energies.cpu().numpy())
    return metrics

if __name__ == '__main__':    
    name = f'sample/{args.date}'
    if not os.path.exists(name):
        os.makedirs(name)

    wandb.init(project=args.project, config=args.__dict__)
    wandb.run.log_code(".")

    energy = get_energy()
    
    buffer = ReplayBuffer(args.buffer_size, 'cpu', energy.log_reward, args.eval_size, data_ndim=energy.data_ndim, beta=args.beta,
                          rank_weight=args.rank_weight, prioritized=args.prioritized)
    
    buffer.load_data("data/md_600_05")
    
    gfn_model = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
            trajectory_length=args.T, clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
            langevin=args.langevin, learned_variance=args.learned_variance,
            partial_energy=args.partial_energy, log_var_range=args.log_var_range,
            pb_scale_range=args.pb_scale_range,
            t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
            conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
            architecture=args.architecture, lgv_layers=args.lgv_layers,
            joint_layers=args.joint_layers, zero_init=args.zero_init, device=args.device, 
            scheduler=args.scheduler, sigma_max=args.sigma_max, sigma_min=args.sigma_min).to(args.device)
    
    gfn_model.load_state_dict(torch.load(f'result/{args.date}/policy_20000.pt'), strict=False)
    gfn_model.eval()
    
    metrics = dict()

    with torch.no_grad():
        metrics.update(eval(energy, buffer, gfn_model))

    wandb.log(metrics) 