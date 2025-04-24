# import torch
# from torch import nn

# import torch

# class TimeEncodingPIS(nn.Module):
#     def __init__(self, harmonics_dim: int, dim: int, hidden_dim: int = 64):
#         super(TimeEncodingPIS, self).__init__()

#         pe = torch.linspace(start=0.1, end=100, steps=harmonics_dim)[None]

#         self.timestep_phase = nn.Parameter(torch.randn(harmonics_dim)[None])

#         self.t_model = nn.Sequential(
#             nn.Linear(2 * harmonics_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim),
#         )
#         self.register_buffer('pe', pe)

#     def forward(self, t: float = None):
#         """
#         Arguments:
#             t: float
#         """
#         t_sin = ((t * self.pe) + self.timestep_phase).sin()
#         t_cos = ((t * self.pe) + self.timestep_phase).cos()
#         t_emb = torch.cat([t_sin, t_cos], dim=-1)
#         return self.t_model(t_emb)


# class StateEncodingPIS(nn.Module):
#     def __init__(self, s_dim: int, hidden_dim: int = 64, s_emb_dim: int = 64):
#         super(StateEncodingPIS, self).__init__()

#         self.x_model = nn.Linear(s_dim, s_emb_dim)

#     def forward(self, s):
#         return self.x_model(s)


# class JointPolicyPIS(nn.Module):
#     def __init__(self, s_dim: int, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = None,
#                  num_layers: int = 2,
#                  zero_init: bool = False):
#         super(JointPolicyPIS, self).__init__()
#         if out_dim is None:
#             out_dim = 2 * s_dim

#         assert s_emb_dim == t_dim, print("Dimensionality of state embedding and time embedding should be the same!")

#         self.model = nn.Sequential(
#             nn.GELU(),
#             *[
#                 nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
#                 for _ in range(num_layers)
#             ],
#             nn.Linear(hidden_dim, out_dim),
#         )

#         if zero_init:
#             self.model[-1].weight.data.fill_(0.0)
#             self.model[-1].bias.data.fill_(0.0)

#     def forward(self, s, t):
#         return self.model(s + t)
    
    
# class FlowModelPIS(nn.Module):
#     def __init__(self, s_dim: int, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1,
#                  num_layers: int = 2,
#                  zero_init: bool = False):
#         super(FlowModelPIS, self).__init__()

#         assert s_emb_dim == t_dim, print("Dimensionality of state embedding and time embedding should be the same!")

#         self.model = nn.Sequential(
#             nn.GELU(),
#             *[
#                 nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
#                 for _ in range(num_layers)
#             ],
#             nn.Linear(hidden_dim, out_dim),
#         )

#         if zero_init:
#             self.model[-1].weight.data.fill_(0.0)
#             self.model[-1].bias.data.fill_(0.0)

#     def forward(self, s, t):
#         return self.model(s + t)

# class LangevinScalingModelPIS(nn.Module):
#     def __init__(self, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1, num_layers: int = 3,
#                  zero_init: bool = False):
#         super(LangevinScalingModelPIS, self).__init__()

#         pe = torch.linspace(start=0.1, end=100, steps=t_dim)[None]

#         self.timestep_phase = nn.Parameter(torch.randn(t_dim)[None])

#         self.lgv_model = nn.Sequential(
#             nn.Linear(2 * t_dim, hidden_dim),
#             *[
#                 nn.Sequential(
#                     nn.GELU(),
#                     nn.Linear(hidden_dim, hidden_dim),
#                 )
#                 for _ in range(num_layers - 1)
#             ],
#             nn.GELU(),
#             nn.Linear(hidden_dim, out_dim)
#         )

#         self.register_buffer('pe', pe)

#         if zero_init:
#             self.lgv_model[-1].weight.data.fill_(0.0)
#             self.lgv_model[-1].bias.data.fill_(0.01)

#     def forward(self, t):
#         t_sin = ((t * self.pe) + self.timestep_phase).sin()
#         t_cos = ((t * self.pe) + self.timestep_phase).cos()
#         t_emb = torch.cat([t_sin, t_cos], dim=-1)
#         return self.lgv_model(t_emb)