import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tbg.tbg.models2 import EGNN_dynamics_AD2_cat


class RNDModel(nn.Module):
    def __init__(self, args, dim, energy):
        super(RNDModel, self).__init__()

        n_particles = dim // 3
        if energy == 'aldp':
            atom_types = np.arange(n_particles)

            atom_types[[0, 2, 3]] = 2
            atom_types[[19, 20, 21]] = 20
            atom_types[[11, 12, 13]] = 12
        elif energy == 'lj13' or energy == 'lj55':
            atom_types = np.zeros(n_particles, dtype=np.int64)
            
        h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))
        
        # Fixed target network: random weights, not updated during training.
        self.target = EGNN_dynamics_AD2_cat(
            n_particles=n_particles,
            device=args.device,
            n_dimension=dim // n_particles,
            h_initial=h_initial,
            hidden_nf=args.hidden_dim,
            act_fn=torch.nn.SiLU(),
            n_layers=args.joint_layers,
            recurrent=True,
            tanh=True,
            attention=True,
            condition_time=True,
            mode="egnn_dynamics",
            agg="sum",
        )
        self.predictor = EGNN_dynamics_AD2_cat(
            n_particles=n_particles,
            device=args.device,
            n_dimension=dim // n_particles,
            h_initial=h_initial,
            hidden_nf=args.hidden_dim,
            act_fn=torch.nn.SiLU(),
            n_layers=args.joint_layers,
            recurrent=True,
            tanh=True,
            attention=True,
            condition_time=True,
            mode="egnn_dynamics",
            agg="sum",
        )
        # self.target = nn.Sequential(
        #     nn.Linear(input_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, feature_dim)
        # )
        # Freeze target network parameters.
        for param in self.target.parameters():
            param.requires_grad = False

        # Predictor network: will be trained to mimic the target.
        # self.predictor = nn.Sequential(
        #     nn.Linear(input_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, feature_dim)
        # )

    def forward(self, x):        
        with torch.no_grad():
            target_feat = self.target(0, x)
        predictor_feat = self.predictor(0, x)

        intrinsic_reward = F.mse_loss(predictor_feat, target_feat, reduction='none')
        return intrinsic_reward.sum(-1)