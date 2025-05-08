import torch
import torchani
import numpy as np
from ase.io import read
from ase.data import chemical_symbols
from torchani.units import hartree2kjoulemol

from .base_set import BaseSet
from .particle_system import interatomic_distance

initial_position = [
    [ 0.4795,  0.2779,  0.2894],
    [ 0.5806,  0.2790,  0.2487],
    [ 0.5969,  0.1772,  0.2134],
    [ 0.6600,  0.2979,  0.3210],
    [ 0.5939,  0.3664,  0.1245],
    [ 0.6878,  0.4476,  0.1117],
    [ 0.4952,  0.3559,  0.0324],
    [ 0.4218,  0.2920,  0.0595],
    [ 0.4938,  0.4201, -0.1004],
    [ 0.5950,  0.4136, -0.1405],
    [ 0.3895,  0.3595, -0.2002],
    [ 0.3978,  0.2509, -0.2034],
    [ 0.2935,  0.3737, -0.1507],
    [ 0.4003,  0.4079, -0.2973],
    [ 0.4695,  0.5678, -0.0967],
    [ 0.5234,  0.6447, -0.1790],
    [ 0.3811,  0.6128, -0.0066],
    [ 0.3488,  0.5495,  0.0651],
    [ 0.3521,  0.7590,  0.0289],
    [ 0.2663,  0.7942, -0.0284],
    [ 0.3572,  0.7771,  0.1362],
    [ 0.4319,  0.8246, -0.0059]
]

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

class ALDP(BaseSet):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        
        molecule = read('data/aldp/aldp.pdb')
        atomic_numbers = molecule.get_atomic_numbers()
        atomic_symbols = [chemical_symbols[z] for z in atomic_numbers]     
        
        self.model = torchani.models.ANI2x().to(self.device)
        self.species = self.model.consts.species_to_tensor(atomic_symbols).unsqueeze(0)
        
        self.mass = torch.tensor(mass, device=self.device).unsqueeze(0)
        target_temperature = 300
        kBT = 1.380649 * 6.02214076 * 1e-3 * target_temperature
        self.beta = 1 / kBT
        
        self.data_ndim = len(atomic_numbers) * 3
        self.initial_position = torch.tensor(initial_position, device=self.device)
        
        self.samples = torch.tensor(np.load(f"data/aldp/aldp.npy"))
        self.samples = self.samples[torch.randperm(self.samples.shape[0], generator=torch.Generator().manual_seed(0))]
        
        if args.method in ['ours', 'mle']:
            self.energy_call_count = args.max_iter_ls * args.teacher_batch_size
        else:
            self.energy_call_count = 0

    def energy(self, x, count=False):
        x = x.reshape(-1, self.data_ndim//3, 3)
        species = self.species.repeat(x.shape[0], 1)
        energies_h = self.model((species, 10*x)).energies
        energies_kJmol = hartree2kjoulemol(energies_h)
        energies = energies_kJmol * self.beta
        if count:
            self.energy_call_count += x.shape[0]
        return energies
    
    def non_reduced_energy(self, x, count=False):
        x = x.reshape(-1, self.data_ndim//3, 3)
        species = self.species.repeat(x.shape[0], 1)
        energies_h = self.model((species, 10*x)).energies
        energies_kJmol = hartree2kjoulemol(energies_h)
        if count:
            self.energy_call_count += x.shape[0]
        return energies_kJmol

    def sample(self, batch_size):
        return self.samples[torch.randperm(batch_size)]

    def interatomic_distance(self, x):
        return interatomic_distance(x, self.data_ndim//3, 3, remove_duplicates=True)