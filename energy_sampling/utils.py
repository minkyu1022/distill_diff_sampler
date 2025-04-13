import random
import numpy as np
import math
import PIL

from metrics.gflownet_losses import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def cal_subtb_coef_matrix(lamda, N):
    """
    diff_matrix: (N+1, N+1)
    0, 1, 2, ...
    -1, 0, 1, ...
    -2, -1, 0, ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    range_vals = torch.arange(N + 1)
    diff_matrix = range_vals - range_vals.view(-1, 1)
    B = np.log(lamda) * diff_matrix
    B[diff_matrix <= 0] = -np.inf
    log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
    coef = torch.exp(B - log_total_lambda)
    return coef


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def dcp(tensor):
    return tensor.detach().cpu()


def gaussian_params(tensor):
    mean, logvar = torch.chunk(tensor, 2, dim=-1)
    return mean, logvar


def fig_to_image(fig):
    fig.canvas.draw()

    return PIL.Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


def get_gfn_optimizer(gfn_model, lr_policy, lr_flow, lr_back, back_model=False, conditional_flow_model=False, use_weight_decay=False, weight_decay=1e-7):
    param_groups = [ {'params': gfn_model.t_model.parameters()},
                     {'params': gfn_model.s_model.parameters()},
                     {'params': gfn_model.joint_model.parameters()},
                     {'params': gfn_model.langevin_scaling_model.parameters()} ]
    if conditional_flow_model:
        param_groups += [ {'params': gfn_model.flow_model.parameters(), 'lr': lr_flow} ]
    else:
        param_groups += [ {'params': [gfn_model.flow_model], 'lr': lr_flow} ]

    if back_model:
        param_groups += [ {'params': gfn_model.back_model.parameters(), 'lr': lr_back} ]

    if use_weight_decay:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy, weight_decay=weight_decay)
    else:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy)
    return gfn_optimizer



def get_gfn_forward_loss(mode, init_state, gfn_model, log_reward, coeff_matrix, exploration_std=None, return_exp=False):
    if mode == 'tb':
        loss = fwd_tb(init_state, gfn_model, log_reward, exploration_std, return_exp=return_exp)
    elif mode == 'tb-avg':
        loss = fwd_tb_avg(init_state, gfn_model, log_reward, exploration_std, return_exp=return_exp)
    elif mode == 'db':
        loss = db(init_state, gfn_model, log_reward, exploration_std)
    elif mode == 'subtb':
        loss = subtb(init_state, gfn_model, log_reward, coeff_matrix, exploration_std)
    elif mode == 'pis':
        loss = pis(init_state, gfn_model, log_reward, exploration_std, return_exp=return_exp)
    return loss



def get_gfn_backward_loss(mode, samples, gfn_model, log_reward, exploration_std=None):
    if mode == 'tb':
        loss = bwd_tb(samples, gfn_model, log_reward, exploration_std)
    elif mode == 'tb-avg':
        loss = bwd_tb_avg(samples, gfn_model, log_reward, exploration_std)
    elif mode == 'mle':
        loss = bwd_mle(samples, gfn_model, log_reward, exploration_std)
    return loss


def get_exploration_std(iter, exploratory, exploration_factor=0.1, exploration_wd=False):
    if exploratory is False:
        return None
    if exploration_wd:
        exploration_std = exploration_factor * max(0, 1. - iter / 5000.)
    else:
        exploration_std = exploration_factor
    expl = lambda x: exploration_std
    return expl


def get_name(args):
    name = ''
    if args.langevin:
        name = f'langevin_'
        if args.langevin_scaling_per_dimension:
            name = f'langevin_scaling_per_dimension_'
    if args.exploratory and (args.exploration_factor is not None):
        if args.exploration_wd:
            name = f'exploration_wd_{args.exploration_factor}_{name}_'
        else:
            name = f'exploration_{args.exploration_factor}_{name}_'

    if args.learn_pb:
        name = f'{name}learn_pb_scale_range_{args.pb_scale_range}_'

    if args.clipping:
        name = f'{name}clipping_lgv_{args.lgv_clip}_gfn_{args.gfn_clip}_'

    if args.mode_fwd == 'subtb':
        mode_fwd = f'subtb_subtb_lambda_{args.subtb_lambda}'
        if args.partial_energy:
            mode_fwd = f'{mode_fwd}_{args.partial_energy}'
    else:
        mode_fwd = args.mode_fwd

    if args.both_ways:
        ways = f'fwd_bwd/fwd_{mode_fwd}_bwd_{args.mode_bwd}'
    elif args.bwd:
        ways = f'bwd/bwd_{args.mode_bwd}'
    else:
        ways = f'fwd/fwd_{mode_fwd}'

    if args.local_search:
        local_search = f'local_search_iter_{args.max_iter_ls}_burn_{args.burn_in}_cycle_{args.ls_cycle}_step_{args.ld_step}_beta_{args.beta}_rankw_{args.rank_weight}_prioritized_{args.prioritized}'
        ways = f'{ways}/{local_search}'

    if args.pis_architectures:
        results = 'results_pis_architectures'
    else:
        results = 'results'

    name = f'{results}/{args.teacher}/{args.energy}/{name}gfn/{ways}/T_{args.T}/tscale_{args.t_scale}/lvr_{args.log_var_range}/'

    name = f'{name}/seed_{args.seed}/'

    return name



def compute_dihedral(positions):
    v = positions[:, :-1] - positions[:, 1:]
    v0 = -v[:, 0]
    v1 = v[:, 2]
    v2 = v[:, 1]

    s0 = torch.sum(v0 * v2, dim=-1, keepdim=True) / torch.sum(
        v2 * v2, dim=-1, keepdim=True
    )
    s1 = torch.sum(v1 * v2, dim=-1, keepdim=True) / torch.sum(
        v2 * v2, dim=-1, keepdim=True
    )

    v0 = v0 - s0 * v2
    v1 = v1 - s1 * v2

    v0 = v0 / torch.norm(v0, dim=-1, keepdim=True)
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)

    x = torch.sum(v0 * v1, dim=-1)
    v3 = torch.cross(v0, v2, dim=-1)
    y = torch.sum(v3 * v1, dim=-1)
    return torch.atan2(y, x)


mass = [
    [1.007947, 1.007947, 1.007947],
    [12.01078, 12.01078, 12.01078],
    [1.007947, 1.007947, 1.007947],
    [1.007947, 1.007947, 1.007947],
    [12.01078, 12.01078, 12.01078],
    [15.99943, 15.99943, 15.99943],
    [14.00672, 14.00672, 14.00672],
    [1.007947, 1.007947, 1.007947],
    [12.01078, 12.01078, 12.01078],
    [1.007947, 1.007947, 1.007947],
    [12.01078, 12.01078, 12.01078],
    [1.007947, 1.007947, 1.007947],
    [1.007947, 1.007947, 1.007947],
    [1.007947, 1.007947, 1.007947],
    [12.01078, 12.01078, 12.01078],
    [15.99943, 15.99943, 15.99943],
    [14.00672, 14.00672, 14.00672],
    [1.007947, 1.007947, 1.007947],
    [12.01078, 12.01078, 12.01078],
    [1.007947, 1.007947, 1.007947],
    [1.007947, 1.007947, 1.007947],
    [1.007947, 1.007947, 1.007947]
]
    
    
initial_position = [
    [ 1.2806,  0.3889,  0.8497],
    [ 0.5782,  0.2985,  0.2534],
    [ 0.4952,  0.3327,  0.2920],
    [ 0.5694,  0.1875,  0.2394],
    [ 0.5977,  0.3603,  0.1156],
    [ 0.7067,  0.4150,  0.0864],
    [ 0.4909,  0.3564,  0.0255],
    [ 0.4185,  0.2898,  0.0445],
    [ 0.4937,  0.4239, -0.1042],
    [ 0.5931,  0.4137, -0.1471],
    [ 0.3908,  0.3517, -0.1943],
    [ 0.3007,  0.3243, -0.1362],
    [ 0.3621,  0.4249, -0.2652],
    [ 0.4361,  0.2563, -0.2293],
    [ 0.4726,  0.5712, -0.0919],
    [ 0.4174,  0.6119,  0.0048],
    [ 0.5149,  0.6437, -0.1841],
    [ 0.5558,  0.5997, -0.2636],
    [ 0.4981,  0.7945, -0.1912],
    [ 0.4882,  0.8224, -0.2902],
    [ 0.4070,  0.8311, -0.1548],
    [ 0.5859,  0.8583, -0.1580]
]

def kabsch(P):
    Q = torch.tensor(initial_position, device=P.device).unsqueeze(0)
    centroid_P = torch.mean(P, dim=-2, keepdims=True)
    centroid_Q = torch.mean(Q, dim=-2, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    H = torch.matmul(p.transpose(-2, -1), q)
    U, S, Vt = torch.linalg.svd(H)

    d = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))
    Vt[d < 0.0, -1] *= -1.0

    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
    return torch.matmul(p, R.transpose(-2, -1))