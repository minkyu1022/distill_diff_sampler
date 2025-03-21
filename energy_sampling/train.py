from plot_utils import *
import argparse
import torch
import os

from utils import set_seed, cal_subtb_coef_matrix, fig_to_image, get_gfn_optimizer, get_gfn_forward_loss, \
    get_gfn_backward_loss, get_exploration_std, get_name
from buffer import ReplayBuffer
from langevin import langevin_dynamics
from models import GFN
from gflownet_losses import *
from energies import *
from evaluations import *
from models.ais import annealed_IS_langevin

import matplotlib.pyplot as plt
from tqdm import trange
from tqdm import tqdm

import wandb


parser = argparse.ArgumentParser(description='GFN Linear Regression')

parser.add_argument('--round', type=int, default="1",
                    help="The number of rounds to run teacher-student distillation.")
parser.add_argument('--teacher', type=str, default="mala",
                    choices=('mala', 'ais'),
                    help="Type of teacher sampler.")
parser.add_argument('--teacher_prior', type=str, default="gaussian",
                    choices=('dirac', 'gaussian'),
                    help="Type of prior distribution.")

parser.add_argument('--lr_policy', type=float, default=1e-3)
parser.add_argument('--lr_flow', type=float, default=1e-2)
parser.add_argument('--lr_back', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--s_emb_dim', type=int, default=64)
parser.add_argument('--t_emb_dim', type=int, default=64)
parser.add_argument('--harmonics_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--epochs', type=int, default=25000)
parser.add_argument('--buffer_size', type=int, default=600000)
parser.add_argument('--T', type=int, default=100)
parser.add_argument('--subtb_lambda', type=int, default=2)
parser.add_argument('--t_scale', type=float, default=5.)
parser.add_argument('--log_var_range', type=float, default=4.)
parser.add_argument('--energy', type=str, default='9gmm',
                    choices=('9gmm', '25gmm', '40gmm', 'hard_funnel', 'easy_funnel', 'many_well_32', 'many_well_64', 'many_well_128', 'many_well_512'))
parser.add_argument('--mode_fwd', type=str, default='None', choices=('tb', 'tb-avg', 'db', 'subtb', "pis"))
parser.add_argument('--mode_bwd', type=str, default='None', choices=('tb', 'tb-avg', 'mle'))
parser.add_argument('--both_ways', action='store_true', default=False)

# For local search
################################################################
parser.add_argument('--local_search', action='store_true', default=False)

# How many iterations to run local search
parser.add_argument('--max_iter_ls', type=int, default=4000)

# How many iterations to burn in before making local search
parser.add_argument('--burn_in', type=int, default=2000)

# How frequently to make local search
parser.add_argument('--ls_cycle', type=int, default=100)

# langevin step size
parser.add_argument('--ld_step', type=float, default=0.01)

parser.add_argument('--ld_schedule', action='store_true', default=False)

# target acceptance rate
parser.add_argument('--target_acceptance_rate', type=float, default=0.574)


# For replay buffer
################################################################
# high beta give steep priorization in reward prioritized replay sampling
parser.add_argument('--beta', type=float, default=1.)

# low rank_weighted give steep priorization in rank-based replay sampling
parser.add_argument('--rank_weight', type=float, default=1e-2)

# three kinds of replay training: random, reward prioritized, rank-based
parser.add_argument('--prioritized', type=str, default="rank", choices=('none', 'reward', 'rank'))
################################################################

parser.add_argument('--bwd', action='store_true', default=False)
parser.add_argument('--exploratory', action='store_true', default=False)

parser.add_argument('--sampling', type=str, default="buffer", choices=('sleep_phase', 'energy', 'buffer'))
parser.add_argument('--langevin', action='store_true', default=False)
parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
parser.add_argument('--conditional_flow_model', action='store_true', default=False)
parser.add_argument('--learn_pb', action='store_true', default=False)
parser.add_argument('--pb_scale_range', type=float, default=0.1)
parser.add_argument('--learned_variance', action='store_true', default=False)
parser.add_argument('--partial_energy', action='store_true', default=False)
parser.add_argument('--exploration_factor', type=float, default=0.1)
parser.add_argument('--exploration_wd', action='store_true', default=False)
parser.add_argument('--clipping', action='store_true', default=False)
parser.add_argument('--lgv_clip', type=float, default=1e2)
parser.add_argument('--gfn_clip', type=float, default=1e4)
parser.add_argument('--zero_init', action='store_true', default=False)
parser.add_argument('--pis_architectures', action='store_true', default=False)
parser.add_argument('--lgv_layers', type=int, default=3)
parser.add_argument('--joint_layers', type=int, default=2)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--use_weight_decay', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
args = parser.parse_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

eval_data_size = 2000
final_eval_data_size = 2000
plot_data_size = 2000
final_plot_data_size = 2000

if args.pis_architectures:
    args.zero_init = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device)

if 'None' in args.mode_bwd:
    args.bwd = False
else:
    args.bwd = True

if args.both_ways and args.bwd:
    args.bwd = False

if args.local_search:
    args.both_ways = True


def get_energy():
    if args.energy == '9gmm':
        energy = NineGaussianMixture(device=device)
    elif args.energy == '25gmm':
        energy = TwentyFiveGaussianMixture(device=device)
    elif args.energy == '40gmm':
        energy = FourtyGaussianMixture(device=device)
    elif args.energy == 'hard_funnel':
        energy = HardFunnel(device=device)
    elif args.energy == 'easy_funnel':
        energy = EasyFunnel(device=device)
    elif args.energy == 'many_well_32':
        energy = ManyWell(device=device, dim=32)
    elif args.energy == 'many_well_64':
        energy = ManyWell(device=device, dim=64)
    elif args.energy == 'many_well_128':
        energy = ManyWell(device=device, dim=128)
    elif args.energy == 'many_well_512':
        energy = ManyWell(device=device, dim=512)
    return energy

def get_prior(dim):
    if args.teacher_prior == 'dirac':
        prior = None
    elif args.teacher_prior == 'gaussian':
        prior = Gaussian(device=device, dim=energy.data_ndim, std=1.0)
    else:
        raise ValueError(f"Invalid prior: {args.teacher_prior}")
    
    return prior

def plot_step(energy, gfn_model, buffer, buffer_ls, name):
    if 'many_well' in args.energy:
        # Sample figures by model
        batch_size = plot_data_size
        samples = gfn_model.sample(batch_size, energy.log_reward)

        vizualizations = viz_many_well(energy, samples)
        fig_samples_x13, ax_samples_x13, fig_kde_x13, ax_kde_x13, fig_contour_x13, ax_contour_x13, fig_samples_x23, ax_samples_x23, fig_kde_x23, ax_kde_x23, fig_contour_x23, ax_contour_x23 = vizualizations

        fig_samples_x13.savefig(f'{name}samplesx13.pdf', bbox_inches='tight')
        fig_samples_x23.savefig(f'{name}samplesx23.pdf', bbox_inches='tight')

        fig_kde_x13.savefig(f'{name}kdex13.pdf', bbox_inches='tight')
        fig_kde_x23.savefig(f'{name}kdex23.pdf', bbox_inches='tight')

        fig_contour_x13.savefig(f'{name}contourx13.pdf', bbox_inches='tight')
        fig_contour_x23.savefig(f'{name}contourx23.pdf', bbox_inches='tight')
        
        return {"visualization/contourx13": wandb.Image(fig_to_image(fig_contour_x13)),
                "visualization/contourx23": wandb.Image(fig_to_image(fig_contour_x23)),
                "visualization/kdex13": wandb.Image(fig_to_image(fig_kde_x13)),
                "visualization/kdex23": wandb.Image(fig_to_image(fig_kde_x23)),
                "visualization/samplesx13": wandb.Image(fig_to_image(fig_samples_x13)),
                "visualization/samplesx23": wandb.Image(fig_to_image(fig_samples_x23))}

    elif energy.data_ndim != 2:
        return {}
    
    elif args.energy == '25gmm':
        batch_size = plot_data_size
        samples = gfn_model.sample(batch_size, energy.log_reward)
        gt_samples = energy.sample(batch_size)

        fig_contour, ax_contour = get_figure(bounds=(-20., 20.))
        fig_kde, ax_kde = get_figure(bounds=(-20., 20.))
        fig_kde_overlay, ax_kde_overlay = get_figure(bounds=(-20., 20.))

        plot_contours(energy.log_reward, ax=ax_contour, bounds=(-20., 20.), n_contour_levels=50, device=device)
        plot_kde(gt_samples, ax=ax_kde_overlay, bounds=(-20., 20.))
        plot_kde(samples, ax=ax_kde, bounds=(-20., 20.))
        plot_samples(samples, ax=ax_contour, bounds=(-20., 20.))
        plot_samples(samples, ax=ax_kde_overlay, bounds=(-20., 20.))

        fig_contour.savefig(f'{name}contour.pdf', bbox_inches='tight')
        fig_kde_overlay.savefig(f'{name}kde_overlay.pdf', bbox_inches='tight')
        fig_kde.savefig(f'{name}kde.pdf', bbox_inches='tight')
        # return None
        return {"visualization/contour": wandb.Image(fig_to_image(fig_contour)),
                "visualization/kde_overlay": wandb.Image(fig_to_image(fig_kde_overlay)),
                "visualization/kde": wandb.Image(fig_to_image(fig_kde))}

    elif args.energy == '40gmm':
        batch_size = plot_data_size
        samples = gfn_model.sample(batch_size, energy.log_reward)
        gt_samples = energy.sample(batch_size)

        fig_contour, ax_contour = get_figure(bounds=(-50., 50.))
        fig_kde, ax_kde = get_figure(bounds=(-50., 50.))
        fig_kde_overlay, ax_kde_overlay = get_figure(bounds=(-50., 50.))

        plot_contours(energy.log_reward, ax=ax_contour, bounds=(-50., 50.), n_contour_levels=50, device=device)
        plot_kde(gt_samples, ax=ax_kde_overlay, bounds=(-50., 50.))
        plot_kde(samples, ax=ax_kde, bounds=(-50., 50.))
        plot_samples(samples, ax=ax_contour, bounds=(-50., 50.))
        plot_samples(samples, ax=ax_kde_overlay, bounds=(-50., 50.))

        fig_contour.savefig(f'{name}contour.pdf', bbox_inches='tight')
        fig_kde_overlay.savefig(f'{name}kde_overlay.pdf', bbox_inches='tight')
        fig_kde.savefig(f'{name}kde.pdf', bbox_inches='tight')
        # return None
        return {"visualization/contour": wandb.Image(fig_to_image(fig_contour)),
                "visualization/kde_overlay": wandb.Image(fig_to_image(fig_kde_overlay)),
                "visualization/kde": wandb.Image(fig_to_image(fig_kde))}
    
    else:
        batch_size = plot_data_size
        samples = gfn_model.sample(batch_size, energy.log_reward)
        gt_samples = energy.sample(batch_size)

        fig_contour, ax_contour = get_figure(bounds=(-13., 13.))
        fig_kde, ax_kde = get_figure(bounds=(-13., 13.))
        fig_kde_overlay, ax_kde_overlay = get_figure(bounds=(-13., 13.))

        plot_contours(energy.log_reward, ax=ax_contour, bounds=(-13., 13.), n_contour_levels=150, device=device)
        plot_kde(gt_samples, ax=ax_kde_overlay, bounds=(-13., 13.))
        plot_kde(samples, ax=ax_kde, bounds=(-13., 13.))
        plot_samples(samples, ax=ax_contour, bounds=(-13., 13.))
        plot_samples(samples, ax=ax_kde_overlay, bounds=(-13., 13.))

        fig_contour.savefig(f'{name}contour.pdf', bbox_inches='tight')
        fig_kde_overlay.savefig(f'{name}kde_overlay.pdf', bbox_inches='tight')
        fig_kde.savefig(f'{name}kde.pdf', bbox_inches='tight')
        # return None
        return {"visualization/contour": wandb.Image(fig_to_image(fig_contour)),
                "visualization/kde_overlay": wandb.Image(fig_to_image(fig_kde_overlay)),
                "visualization/kde": wandb.Image(fig_to_image(fig_kde))}


def eval_step(eval_data, energy, gfn_model, final_eval=False):
    gfn_model.eval()
    metrics = dict()
    if final_eval:
        init_state = torch.zeros(final_eval_data_size, energy.data_ndim).to(device)
        samples, metrics['final_eval/log_Z_IS'], metrics['final_eval/ELBO'], metrics[
            'final_eval/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, energy.log_reward)
    else:
        init_state = torch.zeros(eval_data_size, energy.data_ndim).to(device)
        samples, metrics['eval/log_Z_IS'], metrics['eval/ELBO'], metrics[
            'eval/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, energy.log_reward)
            
    if eval_data is None:
        log_elbo = None
        sample_based_metrics = None
    else:
        if final_eval:
            metrics['final_eval/mean_log_likelihood'] = 0. if args.mode_fwd == 'pis' else mean_log_likelihood(eval_data,
                                                                                                              gfn_model,
                                                                                                              energy.log_reward)
            metrics['final_eval/EUBO'] = EUBO(eval_data, gfn_model, energy.log_reward)
        else:
            metrics['eval/mean_log_likelihood'] = 0. if args.mode_fwd == 'pis' else mean_log_likelihood(eval_data,
                                                                                                        gfn_model,
                                                                                                        energy.log_reward)
            metrics['eval/EUBO'] = EUBO(eval_data, gfn_model, energy.log_reward)
        metrics.update(get_sample_metrics(samples, eval_data, final_eval))
    gfn_model.train()
    return metrics


def train_step(energy, gfn_model, gfn_optimizer, it, exploratory, buffer, buffer_ls, exploration_factor, exploration_wd):
    gfn_model.zero_grad()

    exploration_std = get_exploration_std(it, exploratory, exploration_factor, exploration_wd)

    if args.both_ways:
        if it % 2 == 0:
            if args.sampling == 'buffer':
                loss, states, _, _, log_r  = fwd_train_step(energy, gfn_model, exploration_std, return_exp=True)
                # buffer.add(states[:, -1],log_r)
            else:
                loss = fwd_train_step(energy, gfn_model, exploration_std)
        else:
            loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, exploration_std, it=it)

    elif args.bwd:
        loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, exploration_std, it=it)
    else:
        # loss = fwd_train_step(energy, gfn_model, exploration_std)
        loss, states, _, _, log_r  = fwd_train_step(energy, gfn_model, exploration_std, return_exp=True)
        buffer.add(states[:, -1],log_r)

    loss.backward()
    gfn_optimizer.step()
    return loss.item()


def fwd_train_step(energy, gfn_model, exploration_std, return_exp=False):
    init_state = torch.zeros(args.batch_size, energy.data_ndim).to(device)
    loss = get_gfn_forward_loss(args.mode_fwd, init_state, gfn_model, energy.log_reward, coeff_matrix,
                                exploration_std=exploration_std, return_exp=return_exp)
    return loss


def bwd_train_step(energy, gfn_model, buffer, buffer_ls, exploration_std=None, it=0):
    if args.sampling == 'sleep_phase':
        samples = gfn_model.sleep_phase_sample(args.batch_size, exploration_std).to(device)
    elif args.sampling == 'energy':
        samples = energy.sample(args.batch_size).to(device)
    elif args.sampling == 'buffer':
        if args.local_search:
            if it % args.ls_cycle < 2:
                samples, rewards = buffer.sample()
                local_search_samples, log_r = langevin_dynamics(samples, energy.log_reward, device, args)
                buffer_ls.add(local_search_samples, log_r)
        
            samples, rewards = buffer_ls.sample()
        else:
            samples, rewards = buffer.sample()

    loss = get_gfn_backward_loss(args.mode_bwd, samples, gfn_model, energy.log_reward,
                                 exploration_std=exploration_std)
    return loss


def train(energy, buffer, buffer_ls, teacher_flow, name, step_offset):
  
    energy = energy
    eval_data = energy.sample(eval_data_size).to(device)

    gfn_model = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
                    trajectory_length=args.T, clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
                    langevin=args.langevin, learned_variance=args.learned_variance,
                    partial_energy=args.partial_energy, log_var_range=args.log_var_range,
                    pb_scale_range=args.pb_scale_range,
                    t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
                    conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                    pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers,
                    joint_layers=args.joint_layers, zero_init=args.zero_init, device=device).to(device)
    
    if teacher_flow is not None:
        # Load the learned log_Z
        flow_data = teacher_flow
        print("Loaded log_Z:", flow_data)
        
        if not isinstance(flow_data, torch.nn.Parameter):
            flow_data = torch.nn.Parameter(flow_data)
        
        gfn_model.flow_model = flow_data


    gfn_optimizer = get_gfn_optimizer(gfn_model, args.lr_policy, args.lr_flow, args.lr_back, args.learn_pb,
                                      args.conditional_flow_model, args.use_weight_decay, args.weight_decay)
    
    print(gfn_model)
    metrics = dict()
            
    
    gfn_model.train()
    
    for i in trange(args.epochs + 1):
        metrics['train/loss'] = train_step(energy, gfn_model, gfn_optimizer, i, args.exploratory,
                                           buffer, buffer_ls, args.exploration_factor, args.exploration_wd)
        if i % 100 == 0:
            metrics.update(eval_step(eval_data, energy, gfn_model, final_eval=False))
            if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
                del metrics['eval/log_Z_learned']
            images = plot_step(energy, gfn_model, buffer, buffer_ls, name)
            metrics.update(images)
            plt.close('all')
            wandb.log(metrics, step=step_offset+i)
            if i % 1000 == 0:
                torch.save(gfn_model.state_dict(), f'{name}model.pt')

    # Modification of "dict to to" error
    eval_results = final_eval(energy, gfn_model)
    eval_results = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in eval_results.items()}
    # Modification of "dict to to" error
    metrics.update(eval_results)
    if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
        if 'eval/log_Z_learned' in metrics:
          del metrics['eval/log_Z_learned']
    torch.save(gfn_model.state_dict(), f'{name}model_final.pt')

        
    print("Energy calls for student: ", energy.energy_call_count)
    
    return gfn_model, step_offset+args.epochs


def final_eval(energy, gfn_model):
    final_eval_data = energy.sample(final_eval_data_size).to(energy.device)
    results = eval_step(final_eval_data, energy, gfn_model, final_eval=True)
    return results


def eval():
    pass

def teacher_sampling(buffer, prior, energy, meta_dynamic=False, student=None):
    if args.teacher == 'mala':
        
        batch_size = 60
        total_num_MCMC_samples = 0

        with tqdm(total=args.buffer_size, desc="MALA sampling") as pbar:
            while total_num_MCMC_samples < args.buffer_size:
                
                if meta_dynamic and student is not None:
                    population = student.sample(batch_size, energy.log_reward)
                else:
                    if prior is None:
                        population = torch.zeros(batch_size, energy.data_ndim).to(device)
                    else:
                        population = prior.sample(batch_size)
                    
                samples, rewards = langevin_dynamics(x=population, log_reward=energy.log_reward, device=device, args=args, meta_dynamic=meta_dynamic)
                
                buffer.add(samples, rewards)
                
                total_num_MCMC_samples += samples.shape[0]
                pbar.update(samples.shape[0])

        flow_data = torch.tensor(energy.gt_logz())
        
    if args.teacher == 'ais':
        
        batch_size = 3000
        iter_teacher = 200
   
        for i in trange(iter_teacher, desc="AIS sampling"):
            if prior is None:
                raise ValueError("Dirac prior is not supported for AIS.")
            else:
                population = prior.sample(batch_size)
                
            samples, rewards, log_Z_est = annealed_IS_langevin(x=population, prior=prior, energy=energy, trajectory_length=1000, batch_size=batch_size, meta_dynamic=meta_dynamic)    
            
            buffer.add(samples, rewards)
            
        flow_data = log_Z_est
        
    return flow_data

if __name__ == '__main__':
    
    name = get_name(args)
    if not os.path.exists(name):
        os.makedirs(name)

    print(f"Energy: {args.energy}")
    print(f"Teacher: {args.teacher}")
    
    energy = get_energy()
    teacher_prior = get_prior(energy.data_ndim)
    
    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    wandb.init(project="GFN Energy", config=config, name=name)
    
    buffer = ReplayBuffer(args.buffer_size, device, energy.log_reward, args.batch_size,
                            data_ndim=energy.data_ndim, beta=args.beta,
                            rank_weight=args.rank_weight, prioritized=args.prioritized)
    buffer_ls = ReplayBuffer(args.buffer_size, device, energy.log_reward, args.batch_size,
                            data_ndim=energy.data_ndim, beta=args.beta,
                            rank_weight=args.rank_weight, prioritized=args.prioritized)

    global_epochs = 0
    
    for i in range(args.round):
        
        print(f"Round {i+1} of {args.round}")
        
        if i == 0:
            teacher_flow = teacher_sampling(buffer, teacher_prior, energy, meta_dynamic=False, student=None)
            student_model, global_epochs = train(energy, buffer, buffer_ls, teacher_flow, name, step_offset=global_epochs)
        else:
            teacher_flow = teacher_sampling(buffer, teacher_prior, energy, meta_dynamic=True, student=student_model)
            student_model, global_epochs = train(energy, buffer, buffer_ls, student_model.flow_model, name, step_offset=global_epochs)
        
    print(f"Total {args.round} rounds completed")
    print("Global epochs : ", global_epochs)
    
    print(f"Total energy calls: {energy.energy_call_count}")
