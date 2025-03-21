from functools import lru_cache as cache

import torch
import numpy as np

from .base_set import BaseSet
    
class AnnealedDensities:
    def __init__(
        self,
        energy_function: BaseSet,
        prior: BaseSet,
        
    ):
        self.energy_function = energy_function
        self.device = energy_function.device
        self.prior = prior

    def energy(self, t, x):

        prior_energy = self.prior.energy(x)
        energy = self.energy_function.energy(x)

        return (1 - t) * prior_energy + t * energy

    def score(self, t, x):

        prior_score = self.prior.score(x)
        target_score = self.energy_function.score(x)

        return (1 - t) * prior_score + t * target_score

class AnnealedEnergy(BaseSet):
    logZ_is_available = False
    can_sample = False

    def __init__(self, density_family: AnnealedDensities, t):
        target_energy = density_family.energy_function
        super().__init__()

        self.annealed_targets = density_family
        self.t = t

    def gt_logz(self):
        raise NotImplementedError

    def energy(self, x):
        return self.annealed_targets.energy(self.t, x)
      
    def sample(self, batch_size):
        del batch_size
        raise NotImplementedError