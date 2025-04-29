import math
import torch
import numpy as np
from typing import Callable, Sequence, Tuple


# ---------------------------------------------------------------------
# 1.  Generic MALA step  (√2–noise convention: Δx = h∇logπ + √(2h)ξ)
# ---------------------------------------------------------------------
def mala_step_sqrt2(
    x: torch.Tensor,
    logp_fn: Callable[[torch.Tensor], torch.Tensor],
    h: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One Metropolis‑adjusted Langevin step (vectorised over particles).

    Parameters
    ----------
    x : (B, d)  starting points
    logp_fn : callable returning *unnormalised* log‑density
    h : float   step size

    Returns
    -------
    x_new : (B, d)
    accept : (B,)  boolean
    """
    x = x.clone().detach().requires_grad_(True)
    logp_x = logp_fn(x)                       # (B,)
    grad_x = torch.autograd.grad(logp_x.sum(), x)[0]  # (B,d)

    noise = torch.randn_like(x)
    y = x + h * grad_x + (2.0 * h) ** 0.5 * noise

    with torch.no_grad():
        # Forward proposal density  q(x→y)
        logq_xy = -((y - x - h * grad_x) ** 2).sum(-1) / (4.0 * h)
        #   plus constant −½d log(4πh) cancels in the ratio

        # Reverse quantities
        y.requires_grad_(True)
        logp_y = logp_fn(y)
        grad_y = torch.autograd.grad(logp_y.sum(), y)[0]
        logq_yx = -((x - y - h * grad_y) ** 2).sum(-1) / (4.0 * h)

        log_accept = logp_y - logp_x + logq_yx - logq_xy
        accept_prob = torch.exp(torch.minimum(log_accept, torch.zeros_like(log_accept)))
        accept = torch.rand_like(accept_prob) < accept_prob
        x_new = torch.where(accept[:, None], y.detach(), x.detach())

    return x_new, accept


# ---------------------------------------------------------------------
# 2.  AIS with MALA transitions
# ---------------------------------------------------------------------
def annealed_IS_with_mala(
    traj_len: int,
    x0: torch.Tensor,
    prior,
    target,
    step_size: float = 1e-2,
    n_mala_per_beta: int = 1,
    beta_schedule: Sequence[float] | None = None,
    z_est: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    Gradient‑guided AIS ― unbiased log‑Z estimate *and* samples.

    Parameters
    ----------
    traj_len : int
        Number of intermediate inverse‑temperature steps (K).
    x0 : (B,d) tensor
        Initial particles drawn from `prior`.
    prior, target : objects exposing `.log_reward(x)`   (unnormalised log‑densities)
    step_size : float
        MALA step size *in the √2 convention*.
    n_mala_per_beta : int
        Number of MALA moves at each β_{k+1}.
    beta_schedule : iterable of floats in [0,1]
        If None, uses torch.linspace(0,1,traj_len+1).
    z_est : bool
        If True, returns log‑Z estimate via log‑sum‑exp of importance weights.

    Returns
    -------
    x : final particle positions (B,d)
    reward : target.log_reward(x)  (B,)
    log_Z_est : scalar  or None
    elbo : scalar  (mean log‑weight, same as ELBO)
    """
    device = x0.device
    batch_size = x0.shape[0]

    if beta_schedule is None:
        beta_schedule = torch.linspace(0.0, 1.0, traj_len + 1, device=device)

    # Pre‑define callable log‑densities for autograd.
    def logp_prior(z):
        return prior.log_reward(z, count=True)

    def logp_target(z):
        return target.log_reward(z, count=True)

    x = x0.clone()
    log_w = torch.zeros(batch_size, device=device)

    for k in range(len(beta_schedule) - 1):
        β0 = beta_schedule[k]
        β1 = beta_schedule[k + 1]
        Δβ = β1 - β0

        # --- incremental importance weight
        log_w += Δβ * (logp_target(x) - logp_prior(x))

        # --- build log‑density at the *new* temperature β1
        def tempered_logp(z):
            return β1 * logp_target(z) + (1.0 - β1) * logp_prior(z)

        # --- MALA moves to decorrelate
        for _ in range(n_mala_per_beta):
            x, _ = mala_step_sqrt2(x, tempered_logp, step_size)

    # Final quantities at β=1
    reward = logp_target(x)
    elbo = log_w.mean()

    log_Z_est = None
    if z_est:
        log_Z_est = torch.logsumexp(log_w, dim=0) - math.log(batch_size)

    return x.detach(), reward.detach(), log_Z_est, elbo.detach()


# ---------------------------------------------------------------------
# 3.  Convenience wrapper to pick step‑size adaptively (optional)
# ---------------------------------------------------------------------
def find_step_size(
    logp_fn: Callable[[torch.Tensor], torch.Tensor],
    x_init: torch.Tensor,
    target_accept: float = 0.55,
    h_init: float = 1e-2,
    max_iter: int = 50,
) -> float:
    """
    Run short pilot chains to pick a step size giving the desired acceptance rate.
    """
    h = h_init
    x = x_init.clone()
    for _ in range(max_iter):
        x_new, accept = mala_step_sqrt2(x, logp_fn, h)
        acc_rate = accept.float().mean().item()
        factor = 1.1 if acc_rate > target_accept else 0.9
        h *= factor
        x = x_new
    return h
