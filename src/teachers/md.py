import torch
from tqdm import trange

from .base_set import BaseSet

class MD(BaseSet):
    def __init__(self, args, energy):   
        self.energy = energy  
        self.gamma = args.gamma   
        self.device = args.device
        self.n_steps = args.n_steps
        self.timestep = args.timestep
        self.rnd_weight = args.rnd_weight
        self.batch_size = args.teacher_batch_size
        
        kBT = 1.380649 * 6.02214076 * 1e-3 * args.temperature
        
        self.beta = 1 / kBT
        
        self.mass = energy.mass
        self.std = torch.sqrt(2 * kBT * self.gamma * self.timestep / self.mass)

    def sample(self, initial_position, expl_model=None):
        positions = []
        rewards = []
        
        position = initial_position.reshape(self.batch_size, -1, 3)
        velocity = torch.zeros_like(position, device=position.device)
        position = position.requires_grad_(True)
        energy = self.energy.non_reduced_energy(position.reshape(self.batch_size, -1))
        force = -torch.autograd.grad(energy.sum(), position)[0]
        
        for i in trange(self.n_steps):
            position, velocity, force = self.step(position, velocity, force, expl_model)
            if i % 100 == 0:
                reward = self.energy.log_reward(position.reshape(self.batch_size, -1))
                positions.append(position.detach().cpu())
                rewards.append(reward.detach().cpu())
            
        positions = torch.stack(positions, dim=0)
        rewards = torch.stack(rewards, dim=0)
        positions = positions.reshape(-1, self.energy.data_ndim)
        rewards = rewards.reshape(-1)
        print(f"{positions.shape[0]} samples collected")
        return positions.detach(), rewards.detach()

    def step(self, position, velocity, force, expl_model=None):
        with torch.no_grad():
            velocity = (
                (1 - self.gamma * self.timestep) * velocity
                + force * self.timestep / self.mass
                + self.std * torch.randn_like(position, device=position.device)
            )
            position = position + velocity * self.timestep

        position = position.requires_grad_(True)
        
        energy = self.energy.non_reduced_energy(position.reshape(self.batch_size, -1), count=True)
        if expl_model is not None:
            energy = energy - self.rnd_weight * expl_model.forward(position.reshape(self.batch_size, -1))
        force = -torch.autograd.grad(energy.sum(), position)[0]
        return position.detach(), velocity.detach(), force.detach()