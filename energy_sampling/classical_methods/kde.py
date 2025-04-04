import torch
import math

def silverman_bandwidth(x_0: torch.Tensor) -> torch.Tensor:
    """
    Computes the diagonal bandwidth matrix H using Silverman's rule of thumb.
    
    Args:
        x_0: Tensor of shape [N, d], the samples used to estimate the density.
        
    Returns:
        H: Tensor of shape [d, d], the diagonal bandwidth matrix.
    """
    N, d = x_0.shape
    stds = x_0.std(dim=0, unbiased=True)
    factor = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0)) * (N ** (-1.0 / (d + 4.0)))
    diag_H = (factor * stds) ** 2
    H = torch.diag(diag_H)
    return H

class KDEEstimator:
    def __init__(self, x_0: torch.Tensor):
        """
        Precomputes all x_0-dependent parts of the KDE.
        
        Args:
            x_0: Tensor of shape [N, d], the samples for the KDE.
        """
        self.x_0 = x_0  # [N, d]
        self.N, self.d = x_0.shape
        # Precompute bandwidth and its related constants
        self.H = silverman_bandwidth(x_0)           # [d, d]
        self.invH = torch.inverse(self.H)           # [d, d]
        self.detH = torch.det(self.H)
        # log normalizing constant: -0.5*d*log(2*pi) - 0.5*log(det(H))
        self.log_const = -0.5 * self.d * math.log(2 * math.pi) - 0.5 * torch.log(self.detH)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes log f(x) = log KDE(x|x_0) for a batch of query points x.
        
        Args:
            x: Tensor of shape [B, d] at which to evaluate the KDE.
        
        Returns:
            log_p: Tensor of shape [B] containing the log-density estimates.
        """
        # Compute pairwise differences between x and the precomputed x_0
        # z shape: [B, N, d]
        z = x[:, None, :] - self.x_0[None, :, :]
        
        # exponent = -0.5 * (z^T * invH * z) computed per pair (x, x_0)
        exponent = -0.5 * torch.einsum('bnd,dd,bnd->bn', z, self.invH, z)
        
        # log kernel values: logK = log_const + exponent, shape [B, N]
        logK = self.log_const + exponent
        
        # Use torch.logsumexp for numerical stability. Subtract log(N) to normalize.
        log_p = torch.logsumexp(logK, dim=1) - math.log(self.N)
        return log_p