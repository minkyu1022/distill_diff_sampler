from functools import cache

import torch
import numpy as np

from .base_set import BaseSet


class AnnealedDensities:
    def __init__(
        self,
        energy_function: BaseSet,
        prior_energy: BaseSet,
    ):
        self.energy_function = energy_function
        self.device = energy_function.device
        self.prior_energy = prior_energy

    def energy(self, times: torch.Tensor, states: torch.Tensor):

        prior_energy = self.prior_energy.energy(states)
        energy = self.energy_function.energy(states)

        return (1 - times) * prior_energy + times * energy

    def score(self, times: torch.Tensor, states: torch.Tensor):

        prior_score = self.prior_energy.score(states)
        target_score = self.energy_function.score(states)

        return (1 - times) * prior_score + times * target_score

class AnnealedEnergy(BaseSet):
    logZ_is_available = False
    can_sample = False

    def __init__(self, density_family: AnnealedDensities, time: float):
        target_energy = density_family.energy_function
        super().__init__()

        self.annealed_targets = density_family
        self._time = time

    def energy(self, states: torch.Tensor):
        return self.annealed_targets.energy(self._time, states)

    def score(self, states: torch.Tensor):
        return self.annealed_targets.score(self._time, states)

    def _generate_sample(self, batch_size: int) -> torch.Tensor:
        raise Exception("Cannot sample from annealed energy")