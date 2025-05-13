import torch
import math
import numpy as np
import torch.nn as nn

from tbg.tbg.models2 import EGNN_dynamics_AD2_cat
from schedule import LinearNoiseSchedule, GeometricNoiseSchedule

logtwopi = math.log(2 * math.pi)

def gaussian_params(tensor):
    mean, logvar = torch.chunk(tensor, 2, dim=-1)
    return mean, logvar

class GFN(nn.Module):
    def __init__(self, dim: int, s_emb_dim: int, hidden_dim: int,
                 harmonics_dim: int, t_dim: int, log_var_range: float = 4.,
                 t_scale: float = 1., langevin: bool = False, learned_variance: bool = True,
                 trajectory_length: int = 100, partial_energy: bool = False,
                 clipping: bool = False, lgv_clip: float = 1e2, gfn_clip: float = 1e4, pb_scale_range: float = 1.,
                 langevin_scaling_per_dimension: bool = True, conditional_flow_model: bool = False,
                 learn_pb: bool = False,
                 architecture: str = 'egnn', lgv_layers: int = 3, joint_layers: int = 2,
                 zero_init: bool = False, device=torch.device('cuda'), 
                 noise_scheduler: str = 'linear', sigma_max: float = 1, sigma_min: float = 0.01, energy: str = 'aldp',
                 time_scheduler: str = 'uniform', epsilon: float = 1e-4, c: float = 10):
        super(GFN, self).__init__()
        self.dim = dim
        self.harmonics_dim = harmonics_dim
        self.t_dim = t_dim
        self.s_emb_dim = s_emb_dim

        self.trajectory_length = trajectory_length
        self.langevin = langevin
        self.learned_variance = learned_variance
        self.partial_energy = partial_energy
        self.t_scale = t_scale
        
        self.time_scheduler = time_scheduler
        self.epsilon = epsilon
        self.c = c

        self.clipping = clipping
        self.lgv_clip = lgv_clip
        self.gfn_clip = gfn_clip

        self.langevin_scaling_per_dimension = langevin_scaling_per_dimension
        self.conditional_flow_model = conditional_flow_model
        self.learn_pb = learn_pb

        self.architecture = architecture
        self.lgv_layers = lgv_layers
        self.joint_layers = joint_layers

        if noise_scheduler == 'linear':
            noise_schedule = LinearNoiseSchedule(t_scale, 1) # we use t_scale as sigma
        elif noise_scheduler == 'geometric':
            noise_schedule = GeometricNoiseSchedule(sigma_max, sigma_min, 1)
        else:
            raise ValueError("Unknown noise scheduler type")
        
        self.noise_schedule = noise_schedule.to(device)
        
        self.log_var_range = log_var_range
        self.device = device
        n_particles = dim // 3

        if energy == 'aldp':
            atom_types = np.arange(n_particles)

            atom_types[[0, 2, 3]] = 2
            atom_types[[19, 20, 21]] = 20
            atom_types[[11, 12, 13]] = 12
        elif energy == 'lj13' or energy == 'lj55':
            atom_types = np.zeros(n_particles, dtype=np.int64)
            
        h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))
        
        if learn_pb:
            self.back_model = EGNN_dynamics_AD2_cat(
                n_particles=n_particles,
                device=self.device,
                n_dimension=dim // n_particles,
                h_initial=h_initial,
                hidden_nf=hidden_dim,
                act_fn=torch.nn.SiLU(),
                n_layers=joint_layers,
                recurrent=True,
                tanh=True,
                attention=True,
                condition_time=True,
                mode="egnn_dynamics",
                agg="sum",
            )
            
        self.pb_scale_range = pb_scale_range

        if self.conditional_flow_model:
            self.flow_model = EGNN_dynamics_AD2_cat(
                n_particles=n_particles,
                device=self.device,
                n_dimension=dim // n_particles,
                h_initial=h_initial,
                hidden_nf=hidden_dim,
                act_fn=torch.nn.SiLU(),
                n_layers=joint_layers,
                recurrent=True,
                tanh=True,
                attention=True,
                condition_time=True,
                mode="egnn_flow",
                agg="sum",
            )
        else:
            self.flow_model = torch.nn.Parameter(torch.tensor(0., device=self.device))

        # if self.langevin:
        #     self.langevin_scaling_model = EGNN_dynamics_AD2_cat(
        #         n_particles=n_particles,
        #         device=self.device,
        #         n_dimension=dim // n_particles,
        #         h_initial=h_initial,
        #         hidden_nf=hidden_dim,
        #         act_fn=torch.nn.SiLU(),
        #         n_layers=joint_layers,
        #         recurrent=True,
        #         tanh=True,
        #         attention=True,
        #         condition_time=True,
        #         mode="egnn_flow",
        #         agg="sum",
        #     )
        
        self.joint_model = EGNN_dynamics_AD2_cat(
            n_particles=n_particles,
            device=self.device,
            n_dimension=dim // n_particles,
            h_initial=h_initial,
            hidden_nf=hidden_dim,
            act_fn=torch.nn.SiLU(),
            n_layers=joint_layers,
            recurrent=True,
            tanh=True,
            attention=True,
            condition_time=True,
            mode="egnn_dynamics",
            agg="sum",
        )
    def split_params(self, tensor, sigma):
        mean, logvar = gaussian_params(tensor)
        if not self.learned_variance:
            logvar = torch.zeros_like(logvar)
        else:
            logvar = torch.tanh(logvar) * self.log_var_range
        return mean, logvar + torch.log(sigma) * 2.

    def predict_next_state(self, s, t, log_r):
        if self.langevin:
            with torch.enable_grad():
                s.requires_grad_(True)
                grad_log_r = torch.autograd.grad(log_r(s, True).sum(), s)[0].detach()
                grad_log_r = torch.nan_to_num(grad_log_r)
                if self.clipping:
                    grad_log_r = torch.clip(grad_log_r, -self.lgv_clip, self.lgv_clip)

        flow = self.flow_model(t, s).squeeze(-1) if self.conditional_flow_model or self.partial_energy else self.flow_model

        s_new = self.joint_model(t, s)
        dummy = torch.zeros_like(s_new)
        s_new = torch.cat((s_new, dummy), dim=-1)
        
        if self.langevin:
            # scale = self.langevin_scaling_model(t, s)
            # s_new[..., :self.dim] += scale * grad_log_r
            # scale = self.langevin_scaling_model(t, s)
            s_new[..., :self.dim] += grad_log_r
        
        if self.clipping:
            s_new = torch.clip(s_new, -self.gfn_clip, self.gfn_clip)
        return s_new, flow.squeeze(-1)

    def get_trajectory_fwd(self, s, exploration_std, log_r, pis=False):
        bsz = s.shape[0]

        logpf = torch.zeros((bsz, self.trajectory_length), device=self.device)
        logpb = torch.zeros((bsz, self.trajectory_length), device=self.device)
        logf = torch.zeros((bsz, self.trajectory_length + 1), device=self.device)
        states = torch.zeros((bsz, self.trajectory_length + 1, self.dim), device=self.device)

        # build arbitrary time schedule
        time_schedule = self.build_time_schedule()
        for i in range(self.trajectory_length):
            # extract start and end times
            t_i = time_schedule[i]
            t_ip1 = time_schedule[i + 1]
            dt = t_ip1 - t_i

            # center and reshape
            # s = s.reshape(s.shape[0], -1, 3)
            # s = s - torch.mean(s, dim=-2, keepdims=True)
            # s = s.reshape(s.shape[0], -1)

            # predict dynamics
            pfs, flow = self.predict_next_state(s, t_i, log_r)
            pf_mean, pflogvars = self.split_params(pfs, self.noise_schedule(t_i))

            # flow term
            logf[:, i] = flow
            if self.partial_energy:
                ref_log_var = np.log(self.t_scale**2 * max(t_i, time_schedule[1]))
                log_p_ref = -0.5 * (logtwopi + ref_log_var + np.exp(-ref_log_var) * (s ** 2)).sum(1)
                logf[:, i] += (1 - t_i) * log_p_ref + t_i * log_r(s)

            # adjust variance for exploration
            if exploration_std is None:
                pflogvars_sample = pflogvars if pis else pflogvars.detach()
            else:
                expl = exploration_std(i)
                if expl <= 0.0:
                    pflogvars_sample = pflogvars.detach()
                else:
                    add_log_var = torch.full_like(pflogvars, np.log(expl / np.sqrt(dt)) * 2)
                    pflogvars_sample = torch.logaddexp(pflogvars, add_log_var) if pis else torch.logaddexp(pflogvars, add_log_var).detach()

            # sample forward
            noise_scale = np.sqrt(dt) * (pflogvars_sample / 2).exp()
            s_ = s + dt * (pf_mean if pis else pf_mean.detach()) + noise_scale * torch.randn_like(s, device=self.device)

            # compute forward log-prob
            noise = ((s_ - s) - dt * pf_mean) / (np.sqrt(dt) * (pflogvars / 2).exp())
            logpf[:, i] = -0.5 * (noise ** 2 + logtwopi + np.log(dt) + pflogvars).sum(1)

            # backward transition
            if self.learn_pb and i > 0:
                pbs = self.back_model(t_ip1, s_)
                dmean, dvar = gaussian_params(pbs)
                back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
                back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
            else:
                back_mean_correction = back_var_correction = torch.ones_like(s_)

            if i > 0:
                drift = self.noise_schedule.brownian_drift(t_ip1, s_)
                back_mean = s_ - drift * dt * back_mean_correction
                back_var = (self.noise_schedule(t_ip1) ** 2) * dt * (t_i / t_ip1) * back_var_correction
                noise_backward = (s - back_mean) / back_var.sqrt()
                logpb[:, i] = -0.5 * (noise_backward ** 2 + logtwopi + back_var.log()).sum(1)

            # update for next step
            s = s_
            states[:, i + 1] = s

        return states, logpf, logpb, logf

    def get_trajectory_bwd(self, s, exploration_std, log_r):
        bsz = s.shape[0]

        logpf = torch.zeros((bsz, self.trajectory_length), device=self.device)
        logpb = torch.zeros((bsz, self.trajectory_length), device=self.device)
        logf = torch.zeros((bsz, self.trajectory_length + 1), device=self.device)
        states = torch.zeros((bsz, self.trajectory_length + 1, self.dim), device=self.device)

        # build arbitrary time schedule
        time_schedule = self.build_time_schedule()
        states[:, -1] = s

        for i in range(self.trajectory_length):
            # current and previous time indices in schedule
            t_curr = time_schedule[self.trajectory_length - i]
            t_prev = time_schedule[self.trajectory_length - i - 1]
            dt = t_curr - t_prev

            # backward diffusion step (except first iteration)
            if i < self.trajectory_length - 1:
                if self.learn_pb:
                    pbs = self.back_model(t_curr, s)
                    dmean, dvar = gaussian_params(pbs)
                    back_mean_corr = 1 + dmean.tanh() * self.pb_scale_range
                    back_var_corr = 1 + dvar.tanh() * self.pb_scale_range
                else:
                    back_mean_corr = back_var_corr = torch.ones_like(s)

                drift = self.noise_schedule.brownian_drift(t_curr, s)
                mean = s - dt * drift * back_mean_corr
                var = ((self.noise_schedule(t_curr) ** 2) * dt * t_prev / t_curr) * back_var_corr
                s_ = mean.detach() + var.sqrt().detach() * torch.randn_like(s, device=self.device)
                noise_backward = (s_ - mean) / var.sqrt()
                logpb[:, self.trajectory_length - i - 1] = -0.5 * (noise_backward ** 2 + logtwopi + var.log()).sum(1)
            else:
                s_ = torch.zeros_like(s)

            # forward dynamics on backward step
            pfs, flow = self.predict_next_state(s_, t_prev, log_r)
            pf_mean, pflogvars = self.split_params(pfs, self.noise_schedule(t_prev))

            logf[:, self.trajectory_length - i - 1] = flow
            if self.partial_energy:
                ref_log_var = np.log(self.t_scale**2 * max(t_prev, time_schedule[1]))
                log_p_ref = -0.5 * (logtwopi + ref_log_var + np.exp(-ref_log_var) * (s ** 2)).sum(1)
                logf[:, self.trajectory_length - i - 1] += (1 - t_prev) * log_p_ref + t_prev * log_r(s)

            # backward forward transition logpf
            noise = ((s - s_) - dt * pf_mean) / (np.sqrt(dt) * (pflogvars / 2).exp())
            logpf[:, self.trajectory_length - i - 1] = -0.5 * (noise ** 2 + logtwopi + np.log(dt) + pflogvars).sum(1)

            # update state
            s = s_
            states[:, self.trajectory_length - i - 1] = s

        return states, logpf, logpb, logf

    def sample(self, batch_size, log_r):
        s = torch.zeros(batch_size, self.dim).to(self.device)
        return self.get_trajectory_fwd(s, None, log_r)[0][:, -1]

    def sleep_phase_sample(self, batch_size, exploration_std):
        s = torch.zeros(batch_size, self.dim).to(self.device)
        return self.get_trajectory_fwd(s, exploration_std, log_r=None)[0][:, -1]

    def forward(self, s, exploration_std=None, log_r=None):
        return self.get_trajectory_fwd(s, exploration_std, log_r)

    def build_time_schedule(self):
        """Return a list of length `trajectory_length+1`:
           uniform, random‐nonuniform or equidistant. """
        N = self.trajectory_length

        if self.time_scheduler == 'uniform':
            return [i * 1.0  / N for i in range(N + 1)]

        elif self.time_scheduler == 'random':
            z = torch.rand(N, device=self.device) * (self.c - 1) + 1
            t = [0.] + list(torch.cumsum(z / z.sum(), 0).cpu())
            return t 
        else:
            raise ValueError(f"Unknown time_scheduler {self.time_scheduler!r}")
