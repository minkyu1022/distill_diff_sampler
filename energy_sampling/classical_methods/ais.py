import torch
import numpy as np
from energies.annealed_energy import AnnealedDensities, AnnealedEnergy

def annealed_IS_with_langevin(traj_len, x_0, prior, target, expl_model=None, z_est=False):
  
  device = target.device
  trajectory_length = traj_len
  batch_size = x_0.shape[0]
  dt = 1/trajectory_length
  
  x = x_0
  
  if expl_model is not None:
    
    for t in torch.linspace(0, 1, trajectory_length)[1:]:
        
      x = x.requires_grad_(True)
      annealed_log_reward = (1-t) * prior.log_reward(x) + t * (target.log_reward(x) + 100 * expl_model.forward(x))
      annealed_score = torch.autograd.grad(annealed_log_reward.sum(), x)[0]
      x_new = x + annealed_score.detach() * dt + np.sqrt(2 * dt) * torch.randn_like(x, device=x.device)
      
      x = x_new.detach()
        
    reward = target.log_reward(x)
    log_Z_est = None
  
  else:
    
    if z_est:
      
      log_w = torch.zeros(batch_size, device=device)
      prev_log_r = prior.log_reward(x)

      for t in torch.linspace(0, 1, trajectory_length)[1:]:
        
        x = x.requires_grad_(True)
        
        current_log_r = (1-t) * prior.log_reward(x) + t * target.log_reward(x)
        log_w += current_log_r - prev_log_r
        
        annealed_score = torch.autograd.grad(current_log_r.sum(), x)[0]
        x_new = x + annealed_score.detach() * dt + np.sqrt(2 * dt) * torch.randn_like(x, device=x.device)
        
        x = x_new.detach()
        
        prev_log_r = (1-t) * prior.log_reward(x) + t * target.log_reward(x)
      
      reward = target.log_reward(x)
      log_Z_est = torch.logsumexp(log_w, dim=0) - torch.log(torch.tensor(batch_size, device=device))
      
    else:
      for t in torch.linspace(0, 1, trajectory_length)[1:]:
        
        x = x.requires_grad_(True)
        current_log_r = (1-t) * prior.log_reward(x) + t * target.log_reward(x)
        annealed_score = torch.autograd.grad(current_log_r.sum(), x)[0]
        x_new = x + annealed_score.detach() * dt + np.sqrt(2 * dt) * torch.randn_like(x, device=x.device)
        
        x = x_new.detach()
      
      reward = target.log_reward(x)
      log_Z_est = None
    
  return x, reward, log_Z_est