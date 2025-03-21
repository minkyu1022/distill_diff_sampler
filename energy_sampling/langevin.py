import torch
import numpy as np
from torch.distributions.normal import Normal
import torch.nn.functional as F

def adjust_ld_step(current_ld_step, current_acceptance_rate, target_acceptance_rate=0.574, adjustment_factor=0.01):
    """
    Adjust the Langevin dynamics step size based on the current acceptance rate.
    
    :param current_ld_step: Current Langevin dynamics step size.
    :param current_acceptance_rate: Current observed acceptance rate.
    :param target_acceptance_rate: Target acceptance rate, default is 0.574.
    :param adjustment_factor: Factor to adjust the ld_step.
    :return: Adjusted Langevin dynamics step size.
    """
    if current_acceptance_rate > target_acceptance_rate:
        return current_ld_step + adjustment_factor * current_ld_step
    else:
        return current_ld_step - adjustment_factor * current_ld_step

def langevin_dynamics(x, log_reward, device, args, meta_dynamic=False):
    accepted_samples = []
    accepted_logr = []
    acceptance_rate_lst = []
    log_r_original = log_reward(x)
    acceptance_count = 0
    acceptance_rate = 0
    total_proposals = 0

    for i in range(args.max_iter_ls):
        x = x.requires_grad_(True)
        
        r_grad_original = torch.autograd.grad(log_reward(x).sum(), x)[0]
        if args.ld_schedule:
            ld_step = args.ld_step if i == 0 else adjust_ld_step(ld_step, acceptance_rate, target_acceptance_rate=args.target_acceptance_rate)
        else:
            ld_step = args.ld_step

        # If using particle-guidance trick
        # _, potential_grad = get_sim_and_gradient(x, pp='cosine')
        
        if meta_dynamic:
            kde_score = compute_kde_gradient(x, bandwidth=1.0)
            drift = r_grad_original.detach() - kde_score
        else:
            drift = r_grad_original.detach()
        
        
        new_x = x + ld_step * drift + np.sqrt(2 * ld_step) * torch.randn_like(x, device=device)
        log_r_new = log_reward(new_x)
        r_grad_new = torch.autograd.grad(log_reward(new_x).sum(), new_x)[0]

        log_q_fwd = -(torch.norm(new_x - x - ld_step * r_grad_original, p=2, dim=1) ** 2) / (4 * ld_step)
        log_q_bck = -(torch.norm(x - new_x - ld_step * r_grad_new, p=2, dim=1) ** 2) / (4 * ld_step)

        log_accept = (log_r_new - log_r_original) + log_q_bck - log_q_fwd
        accept_mask = torch.rand(x.shape[0], device=device) < torch.exp(torch.clamp(log_accept, max=0))
        acceptance_count += accept_mask.sum().item()
        total_proposals += x.shape[0]

        x = x.detach()
        # After burn-in process
        if i > args.burn_in:
            accepted_samples.append(new_x[accept_mask])
            accepted_logr.append(log_r_new[accept_mask])
        x[accept_mask] = new_x[accept_mask]
        log_r_original[accept_mask] = log_r_new[accept_mask]

        if i % 5 == 0:
            acceptance_rate = acceptance_count / total_proposals
            if i>args.burn_in:
                acceptance_rate_lst.append(acceptance_rate)
            acceptance_count = 0
            total_proposals = 0

    return torch.cat(accepted_samples, dim=0), torch.cat(accepted_logr, dim=0)

def compute_kde_gradient(x, bandwidth=1.0):
    """
    Compute gradient of log density using Kernel Density Estimation with RBF kernel.
    
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim]
        bandwidth (float): Bandwidth parameter for RBF kernel
    
    Returns:
        torch.Tensor: Gradient of log density of shape [batch_size, dim]
    """
    batch_size = x.shape[0]
    
    # Compute pairwise distances
    x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)  # [batch_size, 1]
    distances = x_norm_sq + x_norm_sq.t() - 2 * torch.mm(x, x.t())  # [batch_size, batch_size]
    
    # Compute kernel values
    kernel_values = torch.exp(-distances / (2 * bandwidth**2))  # [batch_size, batch_size]
    
    # Compute differences between all pairs
    x_diff = x.unsqueeze(1) - x.unsqueeze(0)  # [batch_size, batch_size, dim]
    
    # Compute weights for each pair
    weights = kernel_values.unsqueeze(-1) / (bandwidth**2)  # [batch_size, batch_size, 1]
    
    # Compute the gradient
    grad_components = weights * x_diff  # [batch_size, batch_size, dim]
    grad = grad_components.sum(dim=1)  # [batch_size, dim]
    
    # Normalize by kernel sum (avoiding self-interaction)
    kernel_sum = (kernel_values - torch.eye(batch_size, device=x.device)).sum(dim=1, keepdim=True)
    grad = grad / (kernel_sum + 1e-8)
    
    return grad

@torch.enable_grad()
def get_reward_and_gradient(x, log_reward):
    x = x.requires_grad_(True)
    log_r_x = log_reward(x)
    log_r_grad = torch.autograd.grad(log_r_x.sum(), x)[0]

    return log_r_x, log_r_grad

@torch.enable_grad()
def get_sim_and_gradient(x, pp=None):
    x = x.requires_grad_(True)
    
    if pp == 'cosine':
        potential = sum_cosine_similarity(x)
    elif pp == 'rbf':
        potential = sum_rbf_kernel(x)
    else:
        return ValueError('Invalid potential function')
        
    potential_grad = torch.autograd.grad(potential, x)[0]
    
    return potential, potential_grad


def langevin_proposal(x, log_r_grad, step_size, meta_dynamic=False):
    
    if meta_dynamic:
        kde_score = compute_kde_gradient(x, bandwidth=1.0)
        drift = log_r_grad.detach() - kde_score
    else:
        drift = log_r_grad.detach()
    
    mean = step_size * drift
    std = np.sqrt(2 * step_size)
    
    x_new = (
        x
        + mean
        + std * torch.randn_like(x, device=x.device)
    ).detach()
    
    return x_new, mean, std


def correction_step(
    old_x, log_r_old, r_grad_old, new_x, log_r_new, r_grad_new, step_size
):
    device = old_x.device

    log_q_fwd = -(torch.norm(-old_x - step_size * r_grad_old, p=2, dim=1) ** 2) / (
        4 * step_size
    )

    log_q_bck = -(
        torch.norm(old_x - new_x - step_size * r_grad_new, p=2, dim=1) ** 2
    ) / (4 * step_size)

    log_accept = (log_r_new - log_r_old) + log_q_bck - log_q_fwd
    accept_mask = torch.rand(old_x.shape[0], device=device) < torch.exp(
        torch.clamp(log_accept, max=0)
    )

    return accept_mask

def one_step_langevin_dynamic(x, log_reward, step_size, do_correct=False, meta_dynamic=False):
    log_r_old, r_grad_old = get_reward_and_gradient(x, log_reward)

    new_x, mean, std = langevin_proposal(x, r_grad_old, step_size, meta_dynamic)

    if do_correct:
        log_r_new, r_grad_new = get_reward_and_gradient(new_x, log_reward)
        accept_mask = correction_step(
            x, log_r_old, r_grad_old, new_x, log_r_new, r_grad_new, step_size
        )
        x[accept_mask] = new_x[accept_mask]
    else:
        x = new_x

    return x.detach()

def sum_cosine_similarity(x):
    x_norm = F.normalize(x, p=2, dim=1)
    cos_sim = torch.mm(x_norm, x_norm.t())
    
    return cos_sim.sum()

def sum_rbf_kernel(x, gamma=1.0):
    x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)
    distances = x_norm_sq + x_norm_sq.t() - 2 * torch.mm(x, x.t())
    rbf_kernel = torch.exp(-gamma * distances)

    return rbf_kernel.sum()