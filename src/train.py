import os
import wandb
import torch
import argparse

from tqdm import trange

from buffer import ReplayBuffer
from models import GFN, RNDModel

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
parser.add_argument('--epochs', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr_rnd', type=float, default=1e-3)
parser.add_argument('--lr_flow', type=float, default=1e-3)
parser.add_argument('--lr_back', type=float, default=5e-4)
parser.add_argument('--lr_policy', type=float, default=5e-4)
parser.add_argument('--max_grad_norm', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--use_weight_decay', action='store_true', default=False)

## GFN
parser.add_argument('--T', type=int, default=100)
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

def eval(name, energy, buffer, gfn_model, final=False):
    eval_dir = 'final_eval' if final else 'eval'
    metrics = dict()
    
    init_states = torch.zeros(args.eval_size, energy.data_ndim).to(args.device)
    gt_samples = energy.sample(args.eval_size).to(args.device)
    
    samples, metrics[f'{eval_dir}/log_Z_IS'], metrics[f'{eval_dir}/ELBO'], metrics[f'{eval_dir}/log_Z_learned'] = log_partition_function(init_states, gfn_model, energy.log_reward)

    metrics[f'{eval_dir}/mean_log_likelihood'] = torch.tensor(0.0, device=args.device) if args.mode_fwd == 'pis' else mean_log_likelihood(gt_samples, gfn_model, energy.log_reward)
    metrics[f'{eval_dir}/EUBO'] = EUBO(gt_samples, gfn_model, energy.log_reward)
    
    metrics.update(get_sample_metrics(samples, gt_samples, final))

    energies = energy.energy(samples)
    gt_energies = energy.energy(gt_samples)
    
    energy_dict = {
        'Student': energies.detach().cpu().numpy(),
        'GT': gt_energies.detach().cpu().numpy()
    }
    
    if args.method in ['ours', 'mle']:
        teacher_samples = buffer.sample_pos(args.eval_size).to(args.device)
        teacher_energies = energy.energy(teacher_samples)
        energy_dict.update({
            'Teacher': teacher_energies.detach().cpu().numpy()
        })

    energy_hist_fig = plot_energy_hist(energy_dict)
    metrics["visualization/energy_hist"] = wandb.Image(fig_to_image(energy_hist_fig))
    
    if args.energy == 'aldp':
        aldp_fig = draw_aldps(samples[:3])    
        metrics["visualization/aldp"] = wandb.Image(fig_to_image(aldp_fig))
    elif args.energy in ['lj13', 'lj55']:
        dist_dict = {
            'Student': energy.interatomic_distance(samples).reshape(-1).detach().cpu().numpy(),
            'GT': energy.interatomic_distance(gt_samples).reshape(-1).detach().cpu().numpy()
        }
        if args.method in ['ours', 'mle']:
            dist_dict.update({
                'Teacher': energy.interatomic_distance(teacher_samples).reshape(-1).detach().cpu().numpy()
            })
        dist_fig = make_interatomic_dist_fig(dist_dict)
        metrics["visualization/dist"] = wandb.Image(fig_to_image(dist_fig))
        
    # if there is no NaN, Inf, etc, then save the samples of student, teacher and gt
    if not torch.isnan(samples).any() and not torch.isinf(samples).any():
        np.save(f'{name}/samples.npy', samples.cpu().numpy())
        np.save(f'{name}/gt_samples.npy', gt_samples.cpu().numpy())
        if args.method in ['ours', 'mle']:
            np.save(f'{name}/teacher_samples.npy', teacher_samples.cpu().numpy())
    # if there is no NaN, Inf, etc, then save the energys of student, teacher and gt
    if not torch.isnan(energies).any() and not torch.isinf(energies).any():
        np.save(f'{name}/energies.npy', energies.cpu().numpy())
        np.save(f'{name}/gt_energies.npy', gt_energies.cpu().numpy())
        if args.method in ['ours', 'mle']:
            np.save(f'{name}/teacher_energies.npy', teacher_energies.cpu().numpy())
    if args.energy in ['lj13', 'lj55']:
        if not torch.isnan(energy.interatomic_distance(samples)).any() and not torch.isinf(energy.interatomic_distance(samples)).any():
            np.save(f'{name}/distances.npy', energy.interatomic_distance(samples).reshape(-1).cpu().numpy())
            np.save(f'{name}/gt_distances.npy', energy.interatomic_distance(gt_samples).reshape(-1).cpu().numpy())
            if args.method in ['ours', 'mle']:
                np.save(f'{name}/teacher_distances.npy', energy.interatomic_distance(teacher_samples).reshape(-1).cpu().numpy())
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

def train(name, energy, buffer, buffer_ls, epoch_offset, log_Z_est=None):
    gfn_losses = []
    rnd_losses = []
    mlls = []
    elbos = []
    eubos = []
    log_Z_learned = []
    log_Z_IS = []
    energy_call_counts = []
    
    gfn_model = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
            trajectory_length=args.T, clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
            langevin=args.langevin, learned_variance=args.learned_variance,
            partial_energy=args.partial_energy, log_var_range=args.log_var_range,
            pb_scale_range=args.pb_scale_range,
            t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
            conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
            architecture=args.architecture, lgv_layers=args.lgv_layers,
            joint_layers=args.joint_layers, zero_init=args.zero_init, device=args.device, 
            scheduler=args.scheduler, sigma_max=args.sigma_max, sigma_min=args.sigma_min, energy=args.energy).to(args.device)
    
    rnd_model = RNDModel(args, energy.data_ndim, args.energy).to(args.device)
    
    if log_Z_est is not None:
        # Load the learned log_Z
        flow_data = log_Z_est
        # print("Loaded log_Z:", flow_data)
        
        if not isinstance(flow_data, torch.nn.Parameter):
            flow_data = torch.nn.Parameter(flow_data)
        
        gfn_model.flow_model = flow_data

    gfn_optimizer = get_gfn_optimizer(args.architecture, gfn_model, args.lr_policy, args.lr_flow, args.lr_back, args.learn_pb,
                                      args.conditional_flow_model, args.use_weight_decay, args.weight_decay)
    
    rnd_optimizer = torch.optim.Adam(rnd_model.predictor.parameters(), lr=args.lr_rnd)

    metrics = dict()
    
    for i in trange(args.epochs + 1):
        gfn_model.train()
        rnd_model.train()

        metrics['train/loss'], metrics['train/rnd_loss'] = train_step(energy, gfn_model, gfn_optimizer, rnd_model, rnd_optimizer, i, args.exploratory,
                                           buffer, buffer_ls, args.exploration_factor, args.exploration_wd)
    
        metrics['train/energy_call_count'] = energy.energy_call_count
        
        if i % 100 == 0:
            gfn_model.eval()
            with torch.no_grad():
                metrics.update(eval(name, energy, buffer, gfn_model))
            if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
                del metrics['eval/log_Z_learned']
            wandb.log(metrics, step=epoch_offset+i)
            
            torch.save(gfn_model.state_dict(), f'{name}/policy.pt')

            gfn_losses.append(metrics['train/loss'])
            rnd_losses.append(metrics['train/rnd_loss'])
            energy_call_counts.append(metrics['train/energy_call_count'])
            
            elbos.append(metrics['eval/ELBO'].item())
            eubos.append(metrics['eval/EUBO'].item())
            log_Z_IS.append(metrics['eval/log_Z_IS'].item())
            mlls.append(metrics['eval/mean_log_likelihood'].item())
            log_Z_learned.append(metrics['eval/log_Z_learned'].item())
            
            if i % 10000 == 0:
                torch.save(gfn_model.state_dict(), f'{name}/policy_{i}.pt')
                torch.save(rnd_model.state_dict(), f'{name}/rnd_{i}.pt')
           

    gfn_model.eval()
    with torch.no_grad():
        metrics.update(eval(name, energy, buffer, gfn_model, True))
    if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
        del metrics['final_eval/log_Z_learned']
    wandb.log(metrics)
    
    torch.save(gfn_model.state_dict(), f'{name}/policy_final.pt')
    torch.save(rnd_model.state_dict(), f'{name}/rnd_final.pt')
    np.save(f'{name}/gfn_losses.npy', np.array(gfn_losses))
    np.save(f'{name}/rnd_losses.npy', np.array(rnd_losses))
    np.save(f'{name}/energy_call_counts.npy', energy_call_counts)
    np.save(f'{name}/log_Z_learned.npy', log_Z_learned)
    np.save(f'{name}/log_Z_IS.npy', log_Z_IS)
    np.save(f'{name}/elbos.npy', elbos)
    np.save(f'{name}/eubos.npy', eubos)
    np.save(f'{name}/mlls.npy', mlls)
    
    return gfn_model, rnd_model, epoch_offset+args.epochs

if __name__ == '__main__':    
    name = f'result/{args.date}'
    if not os.path.exists(name):
        os.makedirs(name)

    wandb.init(project=args.project, config=args.__dict__)
    wandb.run.log_code(".")

    energy = get_energy()
    teacher = get_teacher()
    
    buffer = ReplayBuffer(args.buffer_size, 'cpu', energy.log_reward, args.batch_size, data_ndim=energy.data_ndim, beta=args.beta,
                          rank_weight=args.rank_weight, prioritized=args.prioritized)
    buffer_ls = ReplayBuffer(args.buffer_size, 'cpu', energy.log_reward, args.batch_size, data_ndim=energy.data_ndim, beta=args.beta,
                          rank_weight=args.rank_weight, prioritized=args.prioritized)
    
    global_epochs = 0
    
    if args.method in ['mle', 'ours'] and args.data_dir:
        log_Z_est = buffer.load_data(args.data_dir, return_flow=True)
    else:
        log_Z_est = None
    
    gfn_model, rnd_model, global_epochs = train(name, energy, buffer, buffer_ls, epoch_offset=global_epochs, log_Z_est=log_Z_est)
    
    if args.method=='ours':
        if args.energy == 'aldp':
            initial_positions = buffer.sample_pos(args.teacher_batch_size).to(args.device)
        if args.energy in ['lj13', 'lj55']:
            prior = Gaussian(args.device, energy.data_ndim, std=args.prior_std)
            initial_positions = prior.sample(args.teacher_batch_size).to(args.device)
        samples, rewards = teacher.sample(initial_positions, expl_model=rnd_model)
        
        buffer.add(samples.detach().cpu(), rewards.detach().cpu())
        np.save(f'{name}/rnd_positions.npy', samples.cpu().numpy())
        
        gfn_model, rnd_model, global_epochs = train(name, energy, buffer, buffer_ls, epoch_offset=global_epochs, log_Z_est=gfn_model.flow_model)
        
    print(f"Total energy calls: {energy.energy_call_count}")