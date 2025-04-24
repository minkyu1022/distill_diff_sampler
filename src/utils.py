import PIL
import math
import scipy
import random
import numpy as np
import mdtraj as md
import networkx as nx
from tqdm import tqdm
from networkx import isomorphism
from bgflow.utils import as_numpy
import networkx.algorithms.isomorphism as iso

from metrics.gflownet_losses import *
from tbg.tbg.utils import create_adjacency_list, find_chirality_centers, compute_chirality_sign, check_symmetry_change


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


def get_gfn_optimizer(architecture, gfn_model, lr_policy, lr_flow, lr_back, back_model=False, conditional_flow_model=False, use_weight_decay=False, weight_decay=1e-7):
    if architecture == 'pis':
        param_groups = [ {'params': gfn_model.t_model.parameters()},
                        {'params': gfn_model.s_model.parameters()},
                        {'params': gfn_model.joint_model.parameters()},
                        {'params': gfn_model.langevin_scaling_model.parameters()} ]
    elif architecture == 'egnn':
        param_groups = [{'params': gfn_model.joint_model.parameters()}]

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


atom_dict = {"C": 0, "H":1, "N":2, "O":3}
traj = md.load("data/aldp.pdb")
topology = traj.topology
atom_types = []
for atom_name in topology.atoms:
    atom_types.append(atom_name.name[0])
atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))
adj_list = torch.from_numpy(np.array([(b.atom1.index, b.atom2.index) for b in topology.bonds], dtype=np.int32))

def align_chiral(samples):
    chirality_centers = find_chirality_centers(adj_list, atom_types)
    reference_signs = compute_chirality_sign(torch.from_numpy(traj.xyz), chirality_centers)
    symmetry_change = check_symmetry_change(samples.reshape(samples.shape[0], -1, 3), chirality_centers, reference_signs)
    print(f"Before correcting symmetry {(~symmetry_change).sum()/len(samples)}")
    samples[symmetry_change] *=-1
    symmetry_change = check_symmetry_change(samples.reshape(samples.shape[0], -1, 3), chirality_centers, reference_signs)
    samples = samples[~symmetry_change]
    return samples

def align_topologies(samples):
    aligned_samples = []
    aligned_idxs = []
    
    samples_np = samples.reshape(samples.shape[0], -1, 3).cpu().numpy()

    for i, sample_np in tqdm(enumerate(samples_np)):   
        aligned_sample, is_isomorphic = align_topology(sample_np, as_numpy(adj_list).tolist())
        if is_isomorphic:
            aligned_samples.append(aligned_sample)
            aligned_idxs.append(i)
    aligned_samples = np.array(aligned_samples)
    print(f"Correct configuration rate {len(aligned_samples)/len(samples)}")
    return torch.from_numpy(aligned_samples).to(samples.device)

def align_topology(sample, reference):
    sample = sample.reshape(-1, 3)
    all_dists = scipy.spatial.distance.cdist(sample, sample)
    adj_list_computed = create_adjacency_list(all_dists, atom_types)
    G_reference = nx.Graph(reference)
    G_sample = nx.Graph(adj_list_computed)
    # not same number of nodes
    if len(G_sample.nodes) != len(G_reference.nodes):
        return sample, False
    for i, atom_type in enumerate(atom_types):
        G_reference.nodes[i]['type']=atom_type
        G_sample.nodes[i]['type']=atom_type
        
    nm = iso.categorical_node_match("type", -1)
    GM = isomorphism.GraphMatcher(G_reference, G_sample, node_match=nm)
    is_isomorphic = GM.is_isomorphic()
    # True
    GM.mapping
    initial_idx = list(GM.mapping.keys())
    final_idx = list(GM.mapping.values())
    sample[initial_idx] = sample[final_idx]
    #print(is_isomorphic)
    return sample, is_isomorphic