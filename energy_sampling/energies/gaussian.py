import numpy as np
import torch

from .base_set import BaseSet

class Gaussian(BaseSet):
    """Guassian energy function with independent noise to each dimension."""

    logZ_is_available = True
    can_sample: bool = True

    def __init__(
        self,
        device,
        dim: int,
        std: float = 1.0,
    ):
        super().__init__()

        self.device = device
        self.dim = dim
        
        self.std = std
        self.logvar = np.log(std**2)

        self.log_two_pi = np.log(2 * np.pi)

        self.log_coeff = (dim / 2) * (self.log_two_pi + self.logvar)

    @property
    def var(self):
        return self.std**2

    @property
    def sigma(self):
        return self.std
      
    def gt_logz(self):
        return 0.0

    def sample(self, batch_size):
        return torch.randn((batch_size, self.dim), device=self.device) * self.sigma

    def energy(self, x):
        assert x.shape[-1] == self.dim
        return 0.5 * (x**2).sum(-1) / self.var + self.log_coeff

    def score(self, x):
        """
        For simple Gaussian, score can be computed without autograd.
        """
        assert x.shape[-1] == self.dim
        return -x / self.var

    def log_prob(self, x):
        assert x.shape[-1] == self.dim
        return -self.energy(x)