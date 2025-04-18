"""
This is Prototype implementation of alanine dipeptide energy function.
It uses openmm API to calculate the energy of the system.

Fast alternatives might include:
    - Exploiting openmm cuda internal implementation.
    - Differentiable molecule force field library (DMFF-JAX)
"""

import torch
import numpy as np

import multiprocessing as mp
from pathlib import Path

from openmm.unit import kilojoule, mole, nanometer, kelvin, picosecond, femtosecond
from openmm.app import AmberPrmtopFile, OBC1
from openmm.openmm import Context, LangevinIntegrator, Platform, System


from .base_set import BaseSet
from .particle_system import interatomic_distance, remove_mean


# Gas constant in kJ/(mol*K)
R = 8.314e-3

DATA_PATH = Path("task/energies/data/aldp")


class AlanineDipeptide(BaseSet):
    """
    Alanine dipeptide energy function.
    It assumes the implicit solvent OBC1, and use amber99 force field.

    This implementation use openmm API and CUDA platform to calculate the energy.
    Note that this implementation is not differentiable.
    """

    logZ_is_available = False

    # Not an exact sample, but from long MD simulation.
    can_sample = True

    def __init__(self, temperature=300, shift_to_minimum=True):
        super().__init__()

        self.temperature = temperature
        self.shift = shift_to_minimum
        self.kBT = R * temperature

        self.spatial_dim = 3
        self.n_particles = 22

        self._init_openmm_context()

        self.minimum_energy_position = torch.load(
            DATA_PATH / "min_energy_position.pt"
        ).squeeze()

        self.approx_sample = torch.load(DATA_PATH / "exact_sample.pt")
        
        self.energy_call_count = 0

    def _init_openmm_context(self):
        # Load the force field file (describing amber99)
        param_topology = AmberPrmtopFile(DATA_PATH / "aldp.prmtop")

        # Create the system (openmm System object)
        self.system: System = param_topology.createSystem(
            implicitSolvent=OBC1,
            constraints=None,
            nonbondedCutoff=None,
            hydrogenMass=None,
        )

        # Create context with platform = CUDA
        self.context = Context(
            self.system,
            LangevinIntegrator(
                300 * kelvin, 1.0 / picosecond, 1.0 * femtosecond
            ),  # Intergrator is not used. It's just dummy.
            Platform.getPlatformByName("CUDA"),
        )

    def energy(self, x: torch.Tensor, count=False):
        if self.shift:
            x = x + self.minimum_energy_position
        
        if count:
            self.energy_call_count += x.shape[0]

        energy = self.energy_and_force(x.detach().cpu())[0]
        return energy.to(x.device)

    def score(self, x: torch.Tensor):
        if self.shift:
            x = x + self.minimum_energy_position

        force = self.energy_and_force(x.detach().cpu())[1]
        return -force.to(x.device)

    def _energy_and_force(self, position: np.ndarray) -> torch.Tensor:
        self.context.setPositions(position)
        state = self.context.getState(getEnergy=True, getForces=True)

        energy = state.getPotentialEnergy().value_in_unit(kilojoule / mole) / self.kBT
        force = (
            state.getForces(asNumpy=True).value_in_unit(kilojoule / mole / nanometer)
            / self.kBT
        )

        return energy, force

    def energy_and_force(self, positions: torch.Tensor) -> torch.Tensor:
        positions: np.ndarray = positions.view(-1, 22, 3).numpy()

        batch_energy, batch_force = zip(*map(self._energy_and_force, positions))

        batch_energy = torch.tensor(batch_energy)
        batch_force = torch.from_numpy(np.stack(batch_force, 0))

        return batch_energy, batch_force

    def _generate_sample(self, batch_size: int) -> torch.Tensor:
        return self.approx_sample[torch.randperm(batch_size)]

    def remove_mean(self, x):
        return remove_mean(x, self.n_particles, self.spatial_dim)

    def interatomic_distance(self, x):
        return interatomic_distance(x, self.n_particles, self.spatial_dim)