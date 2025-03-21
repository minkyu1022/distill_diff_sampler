import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from models.boltzmann_flow import BoltzmannFlow, train_boltzmann_flow
from energies.many_well import ManyWell
from energies.gaussian import Gaussian
from energies.nine_gmm import NineGaussianMixture
from energies.twenty_five_gmm import TwentyFiveGaussianMixture
from energies.hard_funnel import HardFunnel
from energies.easy_funnel import EasyFunnel
from energies.fourty_gmm import FourtyGaussianMixture

def get_energy(energy_name, device):
    if energy_name == 'many_well_32':
        return ManyWell(device=device, dim=32)
    elif energy_name == 'many_well_64':
        return ManyWell(device=device, dim=64)
    elif energy_name == 'many_well_128':
        return ManyWell(device=device, dim=128)
    elif energy_name == 'many_well_512':
        return ManyWell(device=device, dim=512)
    elif energy_name == '9gmm':
        return NineGaussianMixture(device=device)
    elif energy_name == '25gmm':
        return TwentyFiveGaussianMixture(device=device)
    elif energy_name == '40gmm':
        return FourtyGaussianMixture(device=device)
    elif energy_name == 'hard_funnel':
        return HardFunnel(device=device)
    elif energy_name == 'easy_funnel':
        return EasyFunnel(device=device)
    else:
        raise ValueError(f"Unknown energy: {energy_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--energy', type=str, default='many_well_32',
                      choices=('many_well_32', 'many_well_64', 'many_well_128', 'many_well_512',
                              '9gmm', '25gmm', '40gmm', 'hard_funnel', 'easy_funnel'))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=10000)
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get energy function
    energy = get_energy(args.energy, device)
    
    # Generate training data using MCMC
    print("Generating training data...")
    train_data = energy.sample(args.num_samples)
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    model = BoltzmannFlow(
        dim=energy.data_ndim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    print("Starting training...")
    train_boltzmann_flow(
        model=model,
        train_loader=train_loader,
        energy_fn=energy,
        optimizer=optimizer,
        num_epochs=args.num_epochs
    )
    
    # Save model
    torch.save(model.state_dict(), f'boltzmann_flow_{args.energy}.pt')
    print(f"Model saved to boltzmann_flow_{args.energy}.pt")

if __name__ == '__main__':
    main() 