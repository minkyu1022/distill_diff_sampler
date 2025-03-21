import torch

from energies.annealed_energy import AnnealedDensities, AnnealedEnergy
from langevin import one_step_langevin_dynamic


@torch.no_grad()
def annealed_IS_langevin(x, prior, energy, trajectory_length, batch_size, meta_dynamic=False):
    annealed_densities = AnnealedDensities(energy, prior)

    device = energy.device
    dt = 1/trajectory_length

    sample = x
    log_w = torch.zeros(batch_size, device=device)
    
    prev_log_r = AnnealedEnergy(annealed_densities, 0).log_reward(sample)

    for t in torch.linspace(0, 1, trajectory_length)[1:]:
        annealed_energy = AnnealedEnergy(annealed_densities, t)
        
        current_log_r = annealed_energy.log_reward(sample)
        log_w += current_log_r - prev_log_r
        
        sample = one_step_langevin_dynamic(
            sample, annealed_energy.log_reward, dt, do_correct=False, meta_dynamic=meta_dynamic
        )
        
        prev_log_r = annealed_energy.log_reward(sample)
    
    reward = energy.log_reward(sample)
    log_Z_est = torch.logsumexp(log_w, dim=0) - torch.log(torch.tensor(batch_size, device=device))

    return sample, reward, log_Z_est