import torch
import numpy as np
from tqdm import trange
from .base_set import BaseSet

class MALA(BaseSet):
    def __init__(self, args, energy):
        self.energy = energy
        self.device = args.device
        self.burn_in = args.burn_in
        self.ld_step = args.ld_step
        self.rnd_weight = args.rnd_weight
        self.max_iter_ls = args.max_iter_ls
        self.ld_schedule = args.ld_schedule
        self.batch_size = args.teacher_batch_size
        self.target_acceptance_rate = args.target_acceptance_rate

    def adjust_ld_step(self, ld_step, acceptance_rate, adjustment_factor=0.01):
        if acceptance_rate > self.target_acceptance_rate:
            return ld_step + adjustment_factor * ld_step
        else:
            return ld_step - adjustment_factor * ld_step

    def sample(self, x, expl_model=None):
        accepted_samples = []
        accepted_logr = []
        acceptance_rate_lst = []
        log_r_original = self.energy.log_reward(x)
        acceptance_count = 0
        acceptance_rate = 0
        total_proposals = 0

        for i in trange(self.max_iter_ls):
            x = x.requires_grad_(True)
            
            log_r = self.energy.log_reward(x, count=True)
            if expl_model is not None:
                log_r = log_r + self.rnd_weight * expl_model(x)
            r_grad_original = torch.autograd.grad(log_r.sum(), x)[0]
            if self.ld_schedule:
                ld_step = self.ld_step if i == 0 else self.adjust_ld_step(ld_step, acceptance_rate)
            else:
                ld_step = self.ld_step
            
            new_x = x + ld_step * r_grad_original.detach() + np.sqrt(2 * ld_step) * torch.randn_like(x, device=self.device)
            log_r_new = self.energy.log_reward(new_x)
            if expl_model is not None:
                log_r_new = log_r_new + self.rnd_weight * expl_model(new_x)
            r_grad_new = torch.autograd.grad(log_r_new.sum(), new_x)[0]

            with torch.no_grad():
                log_q_fwd = -(torch.norm(new_x - x - ld_step * r_grad_original, p=2, dim=1) ** 2) / (4 * ld_step)
                log_q_bck = -(torch.norm(x - new_x - ld_step * r_grad_new, p=2, dim=1) ** 2) / (4 * ld_step)

                log_accept = (log_r_new - log_r_original) + log_q_bck - log_q_fwd
                accept_mask = torch.rand(x.shape[0], device=self.device) < torch.exp(torch.clamp(log_accept, max=0))
                acceptance_count += accept_mask.sum().item()
                total_proposals += x.shape[0]

                x = x.detach()
                # After burn-in process
                if i > self.burn_in:
                    accepted_samples.append(new_x[accept_mask].detach().cpu())
                    accepted_logr.append(log_r_new[accept_mask].detach().cpu())
                x[accept_mask] = new_x[accept_mask]
                log_r_original[accept_mask] = log_r_new[accept_mask]

                if i % 5 == 0:
                    acceptance_rate = acceptance_count / total_proposals
                    if i>self.burn_in:
                        acceptance_rate_lst.append(acceptance_rate)
                    acceptance_count = 0
                    total_proposals = 0

        sample = torch.cat(accepted_samples, dim=0)
        log_r = self.energy.log_reward(sample)
        print(f"{sample.shape[0]} samples accepted")
        return sample, log_r