import torch
import numpy as np
import bgflow as bg
from .base_set import BaseSet
from bgmol.datasets import Ala2TSF300
from bgflow import TORSIONS, FIXED, BONDS, ANGLES


def load_system():
    dataset = Ala2TSF300(root='data', read=True)
    
    system = dataset.system
    coordinates = dataset.coordinates
    dim_cartesian = len(system.rigid_block) * 3 - 6
    energy_model = dataset.get_energy_model(n_workers=1)
    
    n_train = len(dataset)//2
    permutation = np.random.permutation(n_train)
    all_data = coordinates.reshape(-1, dataset.dim)
    training_data = torch.tensor(all_data[permutation])
    test_data = torch.tensor(all_data[permutation + n_train])
    
    return {
        'dataset': dataset,
        'system': system,
        'energy_model': energy_model,
        'training_data': training_data,
        'test_data': test_data,
        'dim_cartesian': dim_cartesian
    }
    
    
def build_model(system, device):
    coordinate_transform = bg.MixedCoordinateTransformation(
        data=system['training_data'],
        z_matrix=system['system'].z_matrix,
        fixed_atoms=system['system'].rigid_block,
        keepdims=system['dim_cartesian'],
        normalize_angles=True,
    )
    
    shape_info = bg.ShapeDictionary.from_coordinate_transform(coordinate_transform)

    builder = bg.BoltzmannGeneratorBuilder(
        shape_info,
        target=system['energy_model'],
        device=device, 
        dtype=torch.float32,
    )

    for i in range(6):
        builder.add_condition(TORSIONS, on=FIXED)
        builder.add_condition(FIXED, on=TORSIONS)
    for i in range(4):
        builder.add_condition(BONDS, on=ANGLES)
        builder.add_condition(ANGLES, on=BONDS)

    builder.add_map_to_ic_domains()
    builder.add_map_to_cartesian(coordinate_transform)
    
    generator = builder.build_generator()
    return generator


class ALDP(BaseSet):
    def __init__(self, device, dim=66):
        super().__init__()

        aldp = load_system()
        self.bg = build_model(aldp, device)
        self.bg.load_state_dict(torch.load("data/aldp.pt"))
        self.bg = self.bg.to(device)
        
        self.device = device

        self.data_ndim = dim
        
        self.energy_call_count = 0

    def gt_logz(self):
        return 1.

    def energy(self, x):
        self.energy_call_count += x.shape[0]
        return self.bg.energy(x).squeeze(-1)

    def sample(self, batch_size):
        return self.bg.sample(batch_size)