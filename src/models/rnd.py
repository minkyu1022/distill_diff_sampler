import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tbg.tbg.models2 import EGNN_dynamics_AD2_cat


class RNDModel(nn.Module):
    def __init__(self, args, dim):
        super(RNDModel, self).__init__()

        n_particles = dim // 3
        if args.energy == 'aldp':
            atom_types = np.arange(n_particles)

            atom_types[[0, 2, 3]] = 2
            atom_types[[19, 20, 21]] = 20
            atom_types[[11, 12, 13]] = 12
        elif args.energy == 'lj13' or args.energy == 'lj55':
            atom_types = np.zeros(n_particles, dtype=np.int64)
            
        h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))
        
        self.target = EGNN_dynamics_AD2_cat(
            n_particles=n_particles,
            device=args.device,
            n_dimension=dim // n_particles,
            h_initial=h_initial,
            hidden_nf=args.target_hidden_dim,
            act_fn=torch.nn.SiLU(),
            n_layers=args.target_layers,
            recurrent=True,
            tanh=True,
            attention=True,
            mode="egnn_dynamics",
            agg="sum",
        )
        self.predictor = EGNN_dynamics_AD2_cat(
            n_particles=n_particles,
            device=args.device,
            n_dimension=dim // n_particles,
            h_initial=h_initial,
            hidden_nf=args.predictor_hidden_dim,
            act_fn=torch.nn.SiLU(),
            n_layers=args.predictor_layers,
            recurrent=True,
            tanh=True,
            attention=True,
            mode="egnn_dynamics",
            agg="sum",
        )

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):        
        with torch.no_grad():
            target_feat = self.target(0, x)
        predictor_feat = self.predictor(0, x)

        intrinsic_reward = F.mse_loss(predictor_feat, target_feat, reduction='none')
        return intrinsic_reward.sum(-1)