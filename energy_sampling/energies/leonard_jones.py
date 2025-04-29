import logging

import numpy as np
import torch

from .base_set import BaseSet
from .particle_system import interatomic_distance, remove_mean


def lennard_jones_energy(r, eps=1.0, rm=1.0):
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


class LennardJonesEnergy(BaseSet):
    logZ_is_available = False
    can_sample = False

    def __init__(
        self,
        spatial_dim: int,
        n_particles: int,
        device: str,
        epsilon: float = 1.0,
        min_radius: float = 1.0,
        oscillator: bool = True,
        oscillator_scale: float = 1.0,
        energy_factor: float = 1.0,
    ):
        super().__init__()

        self.device = device
        
        self.spatial_dim = spatial_dim
        self.n_particles = n_particles
        self.data_ndim = self.spatial_dim * self.n_particles

        self.epsilon = epsilon
        self.min_radius = min_radius

        self.oscillator = oscillator
        self.oscillator_scale = oscillator_scale

        self.energy_factor = energy_factor
        
        self.energy_call_count = 0

    def energy(self, x: torch.Tensor, count=False):
        assert x.shape[-1] == self.ndim
        
        if count:
            self.energy_call_count += x.shape[0]
        
        # dists is a tensor of shape [..., n_particles * (n_particles - 1) // 2]
        dists = interatomic_distance(x, self.n_particles, self.spatial_dim)

        lj_energies = lennard_jones_energy(dists, self.epsilon, self.min_radius)

        # Each interaction is counted twice
        lj_energies = lj_energies.sum(dim=-1) * self.energy_factor * 2.0

        if self.oscillator:
            x = remove_mean(x, self.n_particles, self.spatial_dim)
            osc_energies = 0.5 * x.pow(2).sum(dim=-1)
            lj_energies = lj_energies + osc_energies * self.oscillator_scale

        return lj_energies

    def sample(self, batch_size: int):
        raise NotImplementedError

    def remove_mean(self, x: torch.Tensor):
        return remove_mean(x, self.n_particles, self.spatial_dim)

    def interatomic_distance(self, x: torch.Tensor):
        return interatomic_distance(
            x, self.n_particles, self.spatial_dim, remove_duplicates=True
        )


class LJ13(LennardJonesEnergy):
    can_sample = True

    def __init__(self, device: str):
        super().__init__(
            spatial_dim=3,
            n_particles=13,
            device=device,
        )
        self.approx_sample = torch.tensor(
            np.load(f"energies/data/LJ13.npy"),
            device=device,
        )
        
    def gt_logz(self):
        raise NotImplementedError

    def sample(self, batch_size: int):
        return self.approx_sample[torch.randperm(batch_size)]


class LJ55(LennardJonesEnergy):
    can_sample = True

    def __init__(self, device: str):
        super().__init__(
            spatial_dim=3,
            n_particles=55,
            device=device,
        )
        self.approx_sample = torch.tensor(
            np.load(f"energies/data/LJ55.npy"),
            device=device,
        )
        
    def gt_logz(self):
        raise NotImplementedError

    def sample(self, batch_size: int):
        return self.approx_sample[torch.randperm(batch_size)]