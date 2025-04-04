import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# RND Module Implementation
# ---------------------------
class RNDModel(nn.Module):
    def __init__(self, input_dim, feature_dim, device=torch.device('cuda')):
        """
        RND module that computes an intrinsic reward as the prediction error
        between a fixed (random) target network and a trainable predictor network.
        Args:
            input_dim (int): Dimension of the input state.
            feature_dim (int): Dimension of the output embedding.
        """
        super(RNDModel, self).__init__()
        
        self.device = device
        
        # Fixed target network: random weights, not updated during training.
        self.target = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        # Freeze target network parameters.
        for param in self.target.parameters():
            param.requires_grad = False

        # Predictor network: will be trained to mimic the target.
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        """
        Computes the intrinsic reward for x as the mean-squared error between
        the predictor’s output and the target network’s output.
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)
        Returns:
            intrinsic_reward (Tensor): A tensor of shape (batch_size,) containing
                the per-sample MSE error.
        """
        with torch.no_grad():
            target_feat = self.target(x)
        predictor_feat = self.predictor(x)
        # Compute per-sample MSE (averaged over feature dimension)
        intrinsic_reward = F.mse_loss(predictor_feat, target_feat)
        return intrinsic_reward
      
def compute_rnd_loss(rnd_module, samples):
    """
    Perform one optimization step on the RND predictor network to minimize the
    mean squared error between its prediction and the target network's output.
    
    Args:
        rnd_module (RNDModule): The RND module instance.
        x_samples (Tensor): Batch of samples used to update the predictor.
        optimizer (torch.optim.Optimizer): Optimizer for rnd_module.predictor parameters.
    
    Returns:
        loss_value (float): The computed MSE loss.
    """
    # Forward pass through predictor and target.
    predictor_feat = rnd_module.predictor(samples)
    with torch.no_grad():
        target_feat = rnd_module.target(samples)
    loss = F.mse_loss(predictor_feat, target_feat)

    return loss