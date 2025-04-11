import math
import torch
import numpy as np
from tqdm import trange

def aldp_md(batch_size, device, target, args, expl_model, initial_position=None):

    mass = [
        [1.007947, 1.007947, 1.007947],
        [12.01078, 12.01078, 12.01078],
        [1.007947, 1.007947, 1.007947],
        [1.007947, 1.007947, 1.007947],
        [12.01078, 12.01078, 12.01078],
        [15.99943, 15.99943, 15.99943],
        [14.00672, 14.00672, 14.00672],
        [1.007947, 1.007947, 1.007947],
        [12.01078, 12.01078, 12.01078],
        [1.007947, 1.007947, 1.007947],
        [12.01078, 12.01078, 12.01078],
        [1.007947, 1.007947, 1.007947],
        [1.007947, 1.007947, 1.007947],
        [1.007947, 1.007947, 1.007947],
        [12.01078, 12.01078, 12.01078],
        [15.99943, 15.99943, 15.99943],
        [14.00672, 14.00672, 14.00672],
        [1.007947, 1.007947, 1.007947],
        [12.01078, 12.01078, 12.01078],
        [1.007947, 1.007947, 1.007947],
        [1.007947, 1.007947, 1.007947],
        [1.007947, 1.007947, 1.007947]
    ]

    initial_position_angstrom = [
        [19.745,  3.573, 20.603],  # HETATM    1  H1  ACE
        [18.882,  3.072, 21.040],  # HETATM    2  CH3 ACE
        [19.299,  2.583, 21.920],  # HETATM    3  H2  ACE
        [18.428,  2.368, 20.342],  # HETATM    4  H3  ACE
        [17.901,  4.069, 21.635],  # HETATM    5  C   ACE
        [16.732,  3.819, 21.870],  # HETATM    6  O   ACE
        [18.376,  5.305, 21.721],  # ATOM      7  N   ALA
        [19.385,  5.264, 21.729],  # ATOM      8  H   ALA
        [17.741,  6.537, 22.185],  # ATOM      9  CA  ALA
        [16.674,  6.456, 21.975],  # ATOM     10  HA  ALA
        [18.128,  6.686, 23.697],  # ATOM     11  CB  ALA
        [17.646,  7.564, 24.127],  # ATOM     12  HB1 ALA
        [19.202,  6.812, 23.836],  # ATOM     13  HB2 ALA
        [17.804,  5.840, 24.302],  # ATOM     14  HB3 ALA
        [18.239,  7.806, 21.423],  # ATOM     15  C   ALA
        [19.120,  7.699, 20.619],  # ATOM     16  O   ALA
        [17.667,  8.950, 21.653],  # HETATM   17  N   NME
        [16.917,  8.869, 22.326],  # HETATM   18  H   NME
        [17.764, 10.096, 20.832],  # HETATM   19  C   NME
        [16.789, 10.514, 20.583],  # HETATM   20  H1  NME
        [18.366, 10.885, 21.283],  # HETATM   21  H2  NME
        [18.270,  9.740, 19.934]   # HETATM   22  H3  NME
    ]
    mass = torch.tensor(mass, device=device)
    if initial_position is not None:
        position = initial_position.reshape(batch_size, -1, 3)
    else:
        position = torch.tensor(initial_position_angstrom, device=device) * 0.1
        position = position.unsqueeze(0).expand(batch_size, -1, -1)
    
    original_kBT = 1.380649 * 6.02214076 * 1e-3 * 300
    original_beta = 1 / original_kBT
    kBT = 1.380649 * 6.02214076 * 1e-3 * args.teacher_temperature
    xi = torch.sqrt(2 * kBT * args.gamma / mass)
    std = xi * math.sqrt(args.timestep)
    velocity = torch.zeros_like(position, device=position.device)
    
    position = position.requires_grad_(True)
    energy = target.energy(position.reshape(batch_size, -1)) / original_beta
    force = -torch.autograd.grad(energy.sum(), position)[0]
    std = np.sqrt(2 * args.gamma * args.timestep)        
            
    def step(position, velocity, force, std):
        with torch.no_grad():
            velocity = (
                (1 - args.gamma * args.timestep) * velocity
                + force * args.timestep / mass
                + std * torch.randn_like(position, device=position.device)
            )
            position = position + velocity * args.timestep
        
        position = position.requires_grad_(True)
        if expl_model is not None:
            energy = target.energy(position.reshape(batch_size, -1)) / original_beta - 50 * expl_model.forward(position.reshape(batch_size, -1)) / original_beta
        else:
            energy = target.energy(position.reshape(batch_size, -1)) / original_beta
        force = -torch.autograd.grad(energy.sum(), position)[0]
        return position, velocity, force
    
    for i in trange(args.n_steps):
        position, velocity, force = step(position.detach(), velocity.detach(), force.detach(), std)
    
    position = position.detach().reshape(batch_size, -1)
    reward = target.log_reward(position)
    return position, reward, None