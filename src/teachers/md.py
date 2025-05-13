import torch
from tqdm import trange

from .base_set import BaseSet

class MD(BaseSet):
    def __init__(self, args, energy):   
        self.energy = energy  
        self.gamma = args.gamma   
        self.device = args.device
        self.burn_in = args.burn_in
        self.ld_step = args.ld_step
        self.rnd_weight = args.rnd_weight
        self.max_iter_ls = args.max_iter_ls
        
        kBT = 1.380649 * 6.02214076 * 1e-3 * args.temperature
        
        self.beta = 1 / kBT
        
        self.mass = energy.mass
        self.std = torch.sqrt(2 * kBT * self.gamma * self.ld_step / self.mass)

    def sample(self, initial_position, expl_model=None):
        positions = []
        rewards = []
        
        position = initial_position.reshape(-1, self.energy.data_ndim//3, 3)
        velocity = torch.zeros_like(position, device=position.device)
        position = position.requires_grad_(True)
        energy = self.energy.non_reduced_energy(position.reshape(-1, self.energy.data_ndim))
        force = -torch.autograd.grad(energy.sum(), position)[0]
        
        for i in trange(self.max_iter_ls):
            position, velocity, force = self.step(position, velocity, force, expl_model)
            reward = self.energy.log_reward(position.reshape(-1, self.energy.data_ndim))
            positions.append(position.detach().cpu())
            rewards.append(reward.detach().cpu())
            
        # stack after burning in first self.burn_in positions and rewards
        positions = torch.stack(positions[self.burn_in:], dim=0)
        rewards = torch.stack(rewards[self.burn_in:], dim=0)
        positions = positions.reshape(-1, self.energy.data_ndim)
        rewards = rewards.reshape(-1)
        print(f"{positions.shape[0]} samples collected")
        return positions.detach(), rewards.detach()

    def step(self, position, velocity, force, expl_model=None):
        with torch.no_grad():
            velocity = (
                (1 - self.gamma * self.ld_step) * velocity
                + force * self.ld_step / self.mass
                + self.std * torch.randn_like(position, device=position.device)
            )
            position = position + velocity * self.ld_step

        position = position.requires_grad_(True)
        
        energy = self.energy.non_reduced_energy(position.reshape(-1, self.energy.data_ndim), count=True)
        if expl_model is not None:
            energy = energy - self.rnd_weight * expl_model.forward(position.reshape(-1, self.energy.data_ndim))
        force = -torch.autograd.grad(energy.sum(), position)[0]
        return position.detach(), velocity.detach(), force.detach()