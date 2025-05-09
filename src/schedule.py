import torch
import torch.nn as nn


class NoiseSchedule(nn.Module):
    def __call__(self, t: float) -> torch.Tensor:
        """
        noise schedule g_t for the SDE of the form dx_t = f(x, t) * dt + g_t * dw_t.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def sigma(self, t: float) -> torch.Tensor:
        return torch.sqrt(self.sigma_squared(t))

    def sigma_squared(self, t: float) -> torch.Tensor:
        """
        Computes the total noise sigma(t)^2 = \int [t to T] g(s)^2 ds.
        It indicates the amount of noise injected into the data between time t and T.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def integrated(self, s: float, e: float) -> torch.Tensor:
        return self.sigma_squared(s) - self.sigma_squared(e)
    
    def brownian_drift(self, t, x):
        g_t = self(t)
        sigma_t = torch.sqrt(self.integrated(0.0, t))
        return ((g_t / sigma_t) ** 2) * x


class LinearNoiseSchedule(NoiseSchedule):
    def __init__(self, std: float, T: float):
        super().__init__()
        self.std = std
        self.T = T

    def __call__(self, t: float) -> torch.Tensor:
        return torch.tensor(self.std, dtype=torch.float32)

    def sigma_squared(self, t: float) -> torch.Tensor:
        return torch.tensor(self.std**2 * (self.T - t), dtype=torch.float32)


class GeometricNoiseSchedule(NoiseSchedule):
    def __init__(self, sigma_max: float, sigma_min: float, T: float):
        super().__init__()
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_diff = sigma_max / sigma_min
        self.T = T

    def __call__(self, t: float) -> torch.Tensor:
        factor = torch.sqrt(2 * torch.log(torch.tensor(self.sigma_diff, dtype=torch.float32)) / self.T)
        exp_term = self.sigma_diff ** (1 - t / self.T)
        return torch.tensor(self.sigma_min, dtype=torch.float32) * factor * torch.tensor(exp_term, dtype=torch.float32)

    def sigma_squared(self, t: float) -> torch.Tensor:
        term = self.sigma_diff ** (2 * (1 - t / self.T)) - 1
        return torch.tensor(self.sigma_min**2 * term, dtype=torch.float32)
