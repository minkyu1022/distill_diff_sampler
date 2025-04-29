import torch
import numpy as np

# def g(t):
#     # Let sigma_d = sigma_max / sigma_min
#     # Then g(t) = sigma_min * sigma_d^t * sqrt{2 * log(sigma_d)}
#     # See Eq 192 in https://arxiv.org/pdf/2206.00364.pdf
#     sigma_min = 0.5
#     sigma_max = 3.0
#     sigma_diff = sigma_max / sigma_min
#     return sigma_min * (sigma_diff**t) * ((2 * np.log(sigma_diff)) ** 0.5)

def annealed_IS_with_langevin(traj_len, x_0, prior, target, expl_model=None, z_est=False):
  
  device = target.device
  trajectory_length = traj_len
  batch_size = x_0.shape[0]
  dt = 1/trajectory_length
  
  x = x_0
  
  if expl_model is not None:
    
    for t in torch.linspace(0, 1, trajectory_length)[1:]:
        
      x = x.requires_grad_(True)
      annealed_log_reward = (1-t) * prior.log_reward(x, count=True) + t * (target.log_reward(x, count=True) + 100 * expl_model.forward(x))
      annealed_score = torch.autograd.grad(annealed_log_reward.sum(), x)[0]
      x_new = x + annealed_score.detach() * dt + np.sqrt(2 * dt) * torch.randn_like(x, device=x.device)
      
      x = x_new.detach()
        
    reward = target.log_reward(x, count=True)
    log_Z_est = None
  
  else:
    
    if z_est:
      
      log_w = torch.zeros(batch_size, device=device)

      prior_log_r, prior_score = log_reward_and_score(prior, x)
      target_log_r, target_score = log_reward_and_score(target, x)
      prev_log_r = prior_log_r
      
      for t in torch.linspace(0, 1, trajectory_length)[1:]:
        
        current_log_r = (1-t) * prior_log_r + t * target_log_r
        log_w += current_log_r - prev_log_r
        
        annealed_score = (1-t) * prior_score + t * target_score
        x_new = x + annealed_score.detach() * dt + np.sqrt(2 * dt) * torch.randn_like(x, device=x.device)
        
        x = x_new
        
        if t == torch.linspace(0, 1, trajectory_length)[-1]: # For the last step, we don't need to compute prev_log_r
          break
        
        prior_log_r, prior_score = log_reward_and_score(prior, x)
        target_log_r, target_score = log_reward_and_score(target, x)
        
        prev_log_r = (1-t) * prior_log_r + t * target_log_r
      
      reward = target.log_reward(x, count=True)
      log_Z_est = torch.logsumexp(log_w, dim=0) - torch.log(torch.tensor(batch_size, device=device))
      # lower_bound = log_w.mean()
      
    else:
      for t in torch.linspace(0, 1, trajectory_length)[1:]:
        
        x = x.requires_grad_(True)
        current_log_r = (1-t) * prior.log_reward(x, count=True) + t * target.log_reward(x, count=True)
        annealed_score = torch.autograd.grad(current_log_r.sum(), x)[0]
        x_new = x + annealed_score.detach() * dt + np.sqrt(2 * dt) * torch.randn_like(x, device=x.device)
        
        x = x_new.detach()
      
      reward = target.log_reward(x, count=True)
      log_Z_est = None
    
  return x, reward, log_Z_est

def log_reward_and_score(energy, x):
    copy_x = x.detach().clone().requires_grad_(True)
    log_reward = energy.log_reward(copy_x, count=True)
    score = torch.autograd.grad(log_reward.sum(), copy_x)[0]
    return log_reward.detach(), score.detach()