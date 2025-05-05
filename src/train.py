import os
import yaml
import wandb
import torch
import argparse

from buffer import ReplayBuffer

from utils import *
from logger import *
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
parser.add_argument('--teacher_batch_size', type=int, default=1000)

# Architecture config
parser.add_argument('--architecture', type=str, default="egnn", choices=('pis', 'egnn'))

## Dimensions
parser.add_argument('--s_emb_dim', type=int, default=64)
parser.add_argument('--t_emb_dim', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--harmonics_dim', type=int, default=64)
parser.add_argument('--target_hidden_dim', type=int, default=32)
parser.add_argument('--predictor_hidden_dim', type=int, default=64)

## Layers
parser.add_argument('--lgv_layers', type=int, default=5)
parser.add_argument('--joint_layers', type=int, default=5)
parser.add_argument('--zero_init', action='store_true', default=False)
parser.add_argument('--target_layers', type=int, default=2)
parser.add_argument('--predictor_layers', type=int, default=3)

# Training config
parser.add_argument('--round', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr_rnd', type=float, default=1e-3)
parser.add_argument('--lr_flow', type=float, default=1e-3)
parser.add_argument('--lr_back', type=float, default=5e-4)
parser.add_argument('--lr_policy', type=float, default=5e-4)
parser.add_argument('--max_grad_norm', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--epochs', type=int, nargs='+', default=[20000])
parser.add_argument('--use_weight_decay', action='store_true', default=False)

## GFN
parser.add_argument('--T', type=int, default=100)
parser.add_argument('--eval_T', type=int, default=500)
parser.add_argument('--t_scale', type=float, default=0.2)
parser.add_argument('--lgv_clip', type=float, default=1e2)
parser.add_argument('--gfn_clip', type=float, default=1e4)
parser.add_argument('--subtb_lambda', type=int, default=2)
parser.add_argument('--rnd_weight', type=float, default=1e9)
parser.add_argument('--sigma_max', type=float, default=0.05)
parser.add_argument('--log_var_range', type=float, default=4.)
parser.add_argument('--sigma_min', type=float, default=0.00001)
parser.add_argument('--pb_scale_range', type=float, default=0.1)
parser.add_argument('--bwd', action='store_true', default=False)
parser.add_argument('--exploration_factor', type=float, default=0.1)
parser.add_argument('--langevin', action='store_true', default=False)
parser.add_argument('--clipping', action='store_true', default=False)
parser.add_argument('--learn_pb', action='store_true', default=False)
parser.add_argument('--both_ways', action='store_true', default=False)
parser.add_argument('--exploratory', action='store_true', default=False)
parser.add_argument('--partial_energy', action='store_true', default=False)
parser.add_argument('--exploration_wd', action='store_true', default=False)
parser.add_argument('--learned_variance', action='store_true', default=False)
parser.add_argument('--conditional_flow_model', action='store_true', default=False)
parser.add_argument('--mode_bwd', type=str, default="tb", choices=('tb', 'tb-avg', 'mle'))
parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
parser.add_argument('--scheduler', type=str, default='linear', choices=('linear', 'geometric'))
parser.add_argument('--mode_fwd', type=str, default="tb", choices=('tb', 'tb-avg', 'db', 'subtb', "pis"))
parser.add_argument('--student_init', type=str, default='reinit', choices=('reinit', 'partialinit','finetune'))
parser.add_argument('--scheduler_type', type=str, default='random', choices=('uniform', 'random', 'equidistant'))

## Local search
parser.add_argument('--ls_cycle', type=int, default=100)
parser.add_argument('--burn_in', type=int, default=15000)
parser.add_argument('--prior_std', type=float, default=1.75)
parser.add_argument('--ld_step', type=float, default=0.00001)
parser.add_argument('--max_iter_ls', type=int, default=20000)
parser.add_argument('--ld_schedule', action='store_true', default=False)
parser.add_argument('--local_search', action='store_true', default=False)
parser.add_argument('--target_acceptance_rate', type=float, default=0.574)

## Replay buffer
parser.add_argument('--beta', type=float, default=1.)
parser.add_argument('--rank_weight', type=float, default=1e-2)
parser.add_argument('--buffer_size', type=int, default=600000)
parser.add_argument('--prioritized', type=str, default="rank", choices=('none', 'reward', 'rank'))
parser.add_argument('--sampling', type=str, default="buffer", choices=('sleep_phase', 'energy', 'buffer'))

# Logging config
parser.add_argument('--checkpoint', type=str, default="")
parser.add_argument('--checkpoint_epoch', type=int, default=0)
parser.add_argument('--eval_size', type=int, default=2000)

args = parser.parse_args()

set_seed(args.seed)

if args.architecture == 'pis':
    args.zero_init = True
if args.both_ways and args.bwd:
    args.bwd = False
if args.local_search:
    args.both_ways = True
coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(args.device)

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

def eval(name, energy, buffer, gfn_model, logging_dict, final=False):
    gfn_model.trajectory_length = args.eval_T
    gfn_model.scheduler_type = 'uniform'
    eval_dir = 'final_eval' if final else 'eval'
    metrics = dict()
    
    with torch.no_grad():
        init_states = torch.zeros(args.eval_size, energy.data_ndim).to(args.device)
        gt_samples = energy.sample(args.eval_size).to(args.device)
        samples, metrics[f'{eval_dir}/log_Z_IS'], metrics[f'{eval_dir}/ELBO'], metrics[f'{eval_dir}/log_Z_learned'] = log_partition_function(init_states, gfn_model, energy.log_reward)
        metrics[f'{eval_dir}/mean_log_likelihood'] = torch.tensor(0.0, device=args.device) if args.mode_fwd == 'pis' else mean_log_likelihood(gt_samples[:200], gfn_model, energy.log_reward)
        metrics[f'{eval_dir}/EUBO'] = EUBO(gt_samples, gfn_model, energy.log_reward)
    
        metrics.update(get_sample_metrics(samples, gt_samples, final))

    energies = energy.energy(samples).detach().cpu().numpy()
    gt_energies = energy.energy(gt_samples).detach().cpu().numpy()
    interatomic_distances = energy.interatomic_distance(samples).reshape(-1).detach().cpu().numpy()
    gt_interatomic_distances = energy.interatomic_distance(gt_samples).reshape(-1).detach().cpu().numpy()
    
    sample_dict = {
        'Student': samples,
        'GT': gt_samples,
    }
    energy_dict = {
        'Student': energies,
        'GT': gt_energies,
    }
    dist_dict = {
        'Student': interatomic_distances,
        'GT': gt_interatomic_distances
    }
    
    if args.method in ['ours', 'mle']:
        teacher_samples = buffer.sample_pos(args.eval_size).to(args.device)
        teacher_energies = energy.energy(teacher_samples).detach().cpu().numpy()
        teacher_interatomic_distances = energy.interatomic_distance(teacher_samples).reshape(-1).detach().cpu().numpy()
        sample_dict.update({
            'Teacher': teacher_samples
        })
        energy_dict.update({
            'Teacher': teacher_energies
        })
        dist_dict.update({
            'Teacher': teacher_interatomic_distances
        })

    energy_hist_fig = plot_energy_hist(energy_dict)
    dist_fig = make_interatomic_dist_fig(dist_dict)
    metrics["visualization/energy_hist"] = wandb.Image(energy_hist_fig)
    metrics["visualization/dist"] = wandb.Image(dist_fig)
    
    if args.energy == 'aldp':
        aldp_fig = draw_aldps(samples[:3])    
        metrics["visualization/aldp"] = wandb.Image(aldp_fig)
    
    if logging_dict['epoch'] % 1000 == 0:
        save_eval(name, sample_dict, energy_dict, dist_dict, logging_dict)
    gfn_model.trajectory_length = args.T
    gfn_model.scheduler_type = args.scheduler_type
    return metrics

def train_step(energy, gfn_model, gfn_optimizer, rnd_model, rnd_optimizer, it, exploratory, buffer, buffer_ls, exploration_factor, exploration_wd):
    gfn_model.zero_grad()
    rnd_model.zero_grad()

    exploration_std = get_exploration_std(it, exploratory, exploration_factor, exploration_wd)

    if args.both_ways:
        if it % 2 == 0:
            if args.sampling == 'buffer':
                loss, states, _, _, log_r  = fwd_train_step(energy, gfn_model, exploration_std, return_exp=True)
                if args.local_search or args.langevin:
                    buffer.add(states[:, -1].detach().cpu(), log_r.detach().cpu())

                rnd_loss = rnd_model.forward(states[:, -1].clone().detach()).mean()
            else:
                loss = fwd_train_step(energy, gfn_model, exploration_std)
        else:
            loss, rnd_loss = bwd_train_step(energy, gfn_model, rnd_model, buffer, buffer_ls, exploration_std, it=it)
    elif args.bwd:
        loss, rnd_loss = bwd_train_step(energy, gfn_model, rnd_model, buffer, buffer_ls, exploration_std, it=it)
    else:
        loss = fwd_train_step(energy, gfn_model, exploration_std)

    loss.backward()
    if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(gfn_model.parameters(), max_norm=args.max_grad_norm)
    gfn_optimizer.step()
    
    if args.mode_bwd == 'tb' and args.both_ways:
        rnd_loss.backward()
        rnd_optimizer.step()
        return loss.item(), rnd_loss.item()
    else:
        return loss.item(), 0

def fwd_train_step(energy, gfn_model, exploration_std, return_exp=False):
    init_state = torch.zeros(args.batch_size, energy.data_ndim, device=args.device)
    loss = get_gfn_forward_loss(args.mode_fwd, init_state, gfn_model, energy.log_reward, coeff_matrix,
                                exploration_std=exploration_std, return_exp=return_exp)
    return loss

def bwd_train_step(energy, gfn_model, rnd_model, buffer, buffer_ls, exploration_std=None, it=0):
    if args.sampling == 'sleep_phase':
        samples = gfn_model.sleep_phase_sample(args.batch_size, exploration_std).to(args.device)
    elif args.sampling == 'energy':
        samples = energy.sample(args.batch_size).to(args.device)
    elif args.sampling == 'buffer': 
        if args.local_search:
            if it % args.ls_cycle < 2:
                samples, rewards = buffer.sample()
                samples = samples.to(args.device).detach()
                rewards = rewards.to(args.device).detach()

                local_search_samples, log_r = teacher.sample(samples)
                buffer_ls.add(local_search_samples, log_r)
        
            samples, rewards = buffer_ls.sample()
            samples = samples.to(args.device).detach()
            rewards = rewards.to(args.device).detach()
        else:
            samples, rewards = buffer.sample()
            samples = samples.to(args.device).detach()
            rewards = rewards.to(args.device).detach()

    loss = get_gfn_backward_loss(args.mode_bwd, samples, gfn_model, energy.log_reward,
                                 exploration_std=exploration_std)

    rnd_loss = rnd_model.forward(samples.clone().detach()).mean()
    
    return loss, rnd_loss

def train(name, energy, buffer, buffer_ls, gfn_model, rnd_model, gfn_optimizer, rnd_optimizer, logging_dict):
    metrics = dict()
    
    while logging_dict['epoch'] < args.epochs[-1]:
        if logging_dict['epoch'] in args.epochs:
            gfn_model.eval()
            rnd_model.eval()
            if args.energy == 'aldp':
                initial_positions = buffer.sample_pos(args.teacher_batch_size).to(args.device)
            if args.energy in ['lj13', 'lj55']:
                prior = Gaussian(args.device, energy.data_ndim, std=args.prior_std)
                initial_positions = prior.sample(args.teacher_batch_size).to(args.device)
            samples, rewards = teacher.sample(initial_positions, expl_model=rnd_model)
            buffer.add(samples.detach().cpu(), rewards.detach().cpu())
            
            flow_data = gfn_model.flow_model
            gfn_model, rnd_model = init_model(args, energy)
            gfn_model.flow_model = flow_data
            gfn_optimizer, rnd_optimizer = init_optimizer(args, gfn_model, rnd_model)
            
        gfn_model.train()
        rnd_model.train()

        metrics['train/gfn_loss'], metrics['train/rnd_loss'] = train_step(energy, gfn_model, gfn_optimizer, rnd_model, rnd_optimizer, logging_dict['epoch'], args.exploratory,
                                           buffer, buffer_ls, args.exploration_factor, args.exploration_wd)
    
        metrics['train/energy_call_count'] = energy.energy_call_count
        
        if logging_dict['epoch'] % 100 == 0:
            print(f"Epoch {logging_dict['epoch']}: GFN loss: {metrics['train/gfn_loss']:.4f}, RND loss: {metrics['train/rnd_loss']:.4f}, Energy call count: {metrics['train/energy_call_count']}")
            gfn_model.eval()
            with torch.no_grad():
                metrics.update(eval(name, energy, buffer, gfn_model, logging_dict))
            wandb.log(metrics, step=logging_dict['epoch'])
            if logging_dict['epoch'] % 1000 == 0:
                save_checkpoint(name, gfn_model, rnd_model, gfn_optimizer, rnd_optimizer, metrics, logging_dict)
            
        logging_dict['epoch'] = logging_dict['epoch'] + 1

    gfn_model.eval()
    with torch.no_grad():
        metrics.update(eval(name, energy, buffer, gfn_model, logging_dict, True))
    wandb.log(metrics)

if __name__ == '__main__':
    # load energy, teacher, and model    
    energy = get_energy()
    teacher = get_teacher()
    buffer = ReplayBuffer(args.buffer_size, 'cpu', energy.log_reward, args.batch_size, data_ndim=energy.data_ndim, beta=args.beta,
                          rank_weight=args.rank_weight, prioritized=args.prioritized)
    buffer_ls = ReplayBuffer(args.buffer_size, 'cpu', energy.log_reward, args.batch_size, data_ndim=energy.data_ndim, beta=args.beta,
                          rank_weight=args.rank_weight, prioritized=args.prioritized)
    gfn_model, rnd_model = init_model(args, energy)
    if args.method == 'ours' and args.data_dir:
        gfn_model.flow_model = torch.nn.Parameter(buffer.load_data(args.data_dir).to(args.device))
    gfn_optimizer, rnd_optimizer = init_optimizer(args, gfn_model, rnd_model)
    
    # load checkpoint
    name = f'result/{args.date}'
    if not os.path.exists(name):
        os.makedirs(name+'/ckpt')
        os.makedirs(name+'/sample')
        os.makedirs(name+'/energy')
        os.makedirs(name+'/dist')
        
    config = vars(args)
    
    if not args.checkpoint:
        logging_dict = {
            'epoch': 0,
            'log_Z_est': None,
            'mlls': [],
            'elbos': [],
            'eubos': [],
            'log_Z_IS': [],
            'gfn_losses': [],
            'rnd_losses': [],
            'log_Z_learned': [],
            'energy_call_counts': []
        }
        with open(f'{name}/config.yml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        path = f'result/{args.checkpoint}/ckpt_{args.checkpoint_epoch}.pth'
        logging_dict = load_checkpoint(path, gfn_model, rnd_model, gfn_optimizer, rnd_optimizer)
    wandb.init(project=args.project, config=config)
    wandb.run.log_code(".")
    
    train(name, energy, buffer, buffer_ls, gfn_model, rnd_model, gfn_optimizer, rnd_optimizer, logging_dict)