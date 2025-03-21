import matplotlib.pyplot as plt

import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet


class TwentyFiveGaussianMixture(BaseSet):
    def __init__(self, device, dim=2):
        super().__init__()
        self.data = torch.tensor([0.0])
        self.device = device

        modes = torch.Tensor([(a, b) for a in [-10, -5, 0, 5, 10] for b in [-10, -5, 0, 5, 10]]).to(self.device)

        nmode = 25
        self.nmode = nmode

        self.data_ndim = dim

        self.gmm = [D.MultivariateNormal(loc=mode.to(self.device),
                                         covariance_matrix=0.3 * torch.eye(self.data_ndim, device=self.device))
                    for mode in modes]
        
        self.energy_call_count = 0

    def gt_logz(self):
        return 0.

    def energy(self, x):
        self.energy_call_count += 1
        log_prob = torch.logsumexp(torch.stack([mvn.log_prob(x) for mvn in self.gmm]), dim=0,
                           keepdim=False) - torch.log(torch.tensor(self.nmode, device=self.device))
        return -log_prob

    def sample(self, batch_size):
        samples = torch.cat([mvn.sample((batch_size // self.nmode,)) for mvn in self.gmm], dim=0).to(self.device)
        return samples

    def viz_pdf(self, fsave="25gmm-density.png"):
        x = torch.linspace(-15, 15, 100).to(self.device)
        y = torch.linspace(-15, 15, 100).to(self.device)
        X, Y = torch.meshgrid(x, y)
        x = torch.stack([X.flatten(), Y.flatten()], dim=1)  # ?

        density = self.unnorm_pdf(x)
        return x, density

    def __getitem__(self, idx):
        del idx
        return self.data[0]
    
# class TwentyFiveGaussianMixture(BaseSet):
#     def __init__(self, device, scale=0.3, dim=2):
#         super().__init__()
#         self.device = device
#         self.data = torch.tensor([0.0])
#         self.data_ndim = 2

#         mean_ls = [
#                 (a, b)
#                 for a in [-10.0, -5.0, 0.0, 5.0, 10.0]
#                 for b in [-10.0, -5.0, 0.0, 5.0, 10.0]
#             ]
#         nmode = len(mean_ls)
#         mean = torch.stack([torch.tensor(xy) for xy in mean_ls])
#         comp = D.Independent(D.Normal(mean.to(self.device), torch.ones_like(mean).to(self.device) * scale), 1)
#         mix = D.Categorical(torch.ones(nmode).to(self.device))
#         self.gmm = MixtureSameFamily(mix, comp)
#         self.data_ndim = dim

#     def gt_logz(self):
#         return 0.

#     def energy(self, x):
#         return -self.gmm.log_prob(x).flatten()

#     def sample(self, batch_size):
#         return self.gmm.sample((batch_size,))

#     def viz_pdf(self, fsave="25gmm-density.png"):
#         x = torch.linspace(-15, 15, 100).to(self.device)
#         y = torch.linspace(-15, 15, 100).to(self.device)
#         X, Y = torch.meshgrid(x, y)
#         x = torch.stack([X.flatten(), Y.flatten()], dim=1)  # ?
        
#         density = self.unnorm_pdf(x)
#         return x, density

#     def __getitem__(self, idx):
#         del idx
#         return self.data[0]
