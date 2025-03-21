import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
# from torchcfm.models import MLP

class BoltzmannFlow(nn.Module):
    def __init__(self, dim, hidden_dim=256, num_layers=4):
        super().__init__()
        self.dim = dim
        
        # Flow matching model
        self.flow_model = MLP(
            dim=dim,
        )
        
        # Flow matcher
        self.flow_matcher = ConditionalFlowMatcher(
            sigma=0.5,  # Noise level
            t0=0.0,     # Start time
            t1=1.0      # End time
        )
        
    def forward(self, x, t):
        """Forward pass of the flow model"""
        return self.flow_model(x, t)
    
    def loss(self, x0, x1, t):
        """Compute the flow matching loss"""
        # Get the flow field
        flow = self.flow_model(x0, t)
        
        # Get the target flow field from the flow matcher
        target_flow = self.flow_matcher.get_flow(x0, x1, t)
        
        # Compute MSE loss
        loss = F.mse_loss(flow, target_flow)
        return loss
    
    def sample(self, num_samples, energy_fn, num_steps=100):
        """Generate samples using the trained flow model"""
        device = next(self.parameters()).device
        
        # Start from noise
        x = torch.randn(num_samples, self.dim, device=device)
        
        # Euler integration
        dt = 1.0 / num_steps
        for t in torch.linspace(1.0, 0.0, num_steps, device=device):
            # Get flow prediction
            flow = self.flow_model(x, t.unsqueeze(0))
            
            # Update x
            x = x + flow * dt
            
            # Add energy-based correction
            if energy_fn is not None:
                with torch.enable_grad():
                    x.requires_grad_(True)
                    energy = energy_fn(x)
                    grad_energy = torch.autograd.grad(energy.sum(), x)[0]
                    x = x - 0.1 * grad_energy * dt
        
        return x

def train_boltzmann_flow(model, train_loader, energy_fn, optimizer, num_epochs=100):
    """Train the Boltzmann flow model"""
    device = next(model.parameters()).device
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            x0 = batch.to(device)
            
            # Sample from target distribution (using energy function)
            with torch.no_grad():
                x1 = energy_fn.sample(x0.shape[0])
            
            # Sample time points
            t = torch.rand(x0.shape[0], 1, device=device)
            
            # Compute loss
            loss = model.loss(x0, x1, t)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}") 