import torch
import torch.nn as nn
from tqdm import tqdm

class ProteinDiffusion(nn.Module):
    def __init__(self, model, n_times=1000, beta_minmax=[1e-4, 2e-2], device='cuda'):
        super(ProteinDiffusion, self).__init__()
        
        self.model = model
        self.n_times = n_times
        self.device = device
        
        # Define linear variance schedule (betas)
        beta_1, beta_T = beta_minmax
        betas = torch.linspace(start=beta_1, end=beta_T, steps=n_times).to(device)
        self.sqrt_betas = torch.sqrt(betas)
        
        # Define alpha for forward diffusion kernel
        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
    
    def extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps"""
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def make_noisy(self, x_zeros, t):
        """Add noise to protein coordinates according to diffusion process"""
        # Create noise with same shape as input coordinates
        epsilon = torch.randn_like(x_zeros).to(self.device)
        
        # Extract coefficients for the given timestep
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)
        
        # Forward diffusion process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
        
        return noisy_sample.detach(), epsilon
    
    def forward(self, x_zeros, seq):
        """Forward pass through the diffusion model"""
        batch_size = x_zeros.shape[0]
        
        # Randomly choose diffusion timesteps
        t = torch.randint(low=0, high=self.n_times, size=(batch_size,)).long().to(self.device)
        
        # Add noise to protein coordinates
        perturbed_coords, epsilon = self.make_noisy(x_zeros, t)
        
        # Predict the noise using our model
        pred_epsilon = self.model(perturbed_coords, seq, t)
        
        return perturbed_coords, epsilon, pred_epsilon
    
    def denoise_at_t(self, x_t, seq, timestep, t):
        """Denoise protein coordinates at specific timestep"""
        batch_size = x_t.shape[0]
        
        # At the last step (t=0), we don't add more noise
        if t > 1:
            z = torch.randn_like(x_t).to(self.device)
        else:
            z = torch.zeros_like(x_t).to(self.device)
        
        # Predict noise at current timestep
        epsilon_pred = self.model(x_t, seq, timestep)
        
        # Extract coefficients for denoising
        alpha = self.extract(self.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas, timestep, x_t.shape)
        
        # Denoise at time t, utilizing predicted noise
        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1-alpha)/sqrt_one_minus_alpha_bar*epsilon_pred) + sqrt_beta*z
        
        return x_t_minus_1
    
    def sample(self, seq, N=1):
        """Generate protein structures from scratch given sequences"""
        # Start from random noise
        batch_size, seq_len, num_aa = seq.shape
        x_t = torch.randn((batch_size, seq_len, 4, 3)).to(self.device)  # 4 backbone atoms per residue
        
        # Iteratively denoise
        for t in tqdm(range(self.n_times-1, -1, -1), desc="Sampling"):
            timestep = torch.tensor([t]).repeat(batch_size).long().to(self.device)
            x_t = self.denoise_at_t(x_t, seq, timestep, t)
        
        return x_t