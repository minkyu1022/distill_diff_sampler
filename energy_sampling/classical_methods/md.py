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
    
    fixed_initial_position = [
        [ 1.2806,  0.3889,  0.8497],
        [ 0.5782,  0.2985,  0.2534],
        [ 0.4952,  0.3327,  0.2920],
        [ 0.5694,  0.1875,  0.2394],
        [ 0.5977,  0.3603,  0.1156],
        [ 0.7067,  0.4150,  0.0864],
        [ 0.4909,  0.3564,  0.0255],
        [ 0.4185,  0.2898,  0.0445],
        [ 0.4937,  0.4239, -0.1042],
        [ 0.5931,  0.4137, -0.1471],
        [ 0.3908,  0.3517, -0.1943],
        [ 0.3007,  0.3243, -0.1362],
        [ 0.3621,  0.4249, -0.2652],
        [ 0.4361,  0.2563, -0.2293],
        [ 0.4726,  0.5712, -0.0919],
        [ 0.4174,  0.6119,  0.0048],
        [ 0.5149,  0.6437, -0.1841],
        [ 0.5558,  0.5997, -0.2636],
        [ 0.4981,  0.7945, -0.1912],
        [ 0.4882,  0.8224, -0.2902],
        [ 0.4070,  0.8311, -0.1548],
        [ 0.5859,  0.8583, -0.1580]
    ]
    
    mass = torch.tensor(mass, device=device).unsqueeze(0)
    if initial_position is not None:
        position = initial_position.reshape(batch_size, -1, 3)
    else:
        position = torch.tensor(fixed_initial_position, device=device) 
        position = position.unsqueeze(0).expand(batch_size, -1, -1)
    
    # position = target.sample(batch_size).reshape(batch_size, -1, 3)
    
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
        
        # if i % 1000 == 0:
        #     np.save(f"./data/bg_md_300/{i}.npy", position.detach().cpu().numpy())
    
    position = position.detach().reshape(batch_size, -1)
    reward = target.log_reward(position)
    return position, reward, None