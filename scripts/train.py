import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys
import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import ProteinDataset, download_example_pdb_files
from src.model import ProteinDenoiser
from src.diffusion import ProteinDiffusion
from src.visualization import plot_training_progress

def train_protein_diffusion(protein_diffusion, train_loader, optimizer, mse_loss, 
                           epochs, device, checkpoint_dir="models", log_dir="logs"):
    """Train the protein diffusion model"""
    
    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create tensorboard writer
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(log_dir, f"proteindiff_{timestamp}"))
    
    print("Starting training of protein diffusion model...")
    
    # Keep track of losses
    epoch_losses = []
    
    protein_diffusion.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            
            # Get data
            coords = batch['coords'].to(device)
            seq = batch['sequence'].to(device)
            
            # Forward pass
            noisy_coords, epsilon, pred_epsilon = protein_diffusion(coords, seq)
            
            # Calculate loss
            loss = mse_loss(pred_epsilon, epsilon)
            epoch_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Log batch loss
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/batch', loss.item(), global_step)
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)
        
        print(f"\tEpoch {epoch + 1} complete! \tDenoising Loss: {avg_epoch_loss:.8f}")
        
        # Log epoch loss
        writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f"proteindiff_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': protein_diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    writer.close()
    print("Training complete!")
    
    # Plot training progress
    plot_training_progress(epoch_losses)
    
    return protein_diffusion, epoch_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ProteinDiff model')
    parser.add_argument('--data_dir', type=str, default='data/pdb_files', help='Directory containing PDB files')
    parser.add_argument('--download_examples', action='store_true', help='Download example PDB files')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--n_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Download example PDB files if requested
    if args.download_examples:
        download_example_pdb_files(args.data_dir)
    
    # Create dataset and dataloader
    dataset = ProteinDataset(args.data_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    print(f"Dataset contains {len(dataset)} protein structures")
    
    # Create model and diffusion process
    hidden_dims = [args.hidden_dim] * args.n_layers
    device = torch.device(args.device)
    
    model = ProteinDenoiser(
        hidden_dims=hidden_dims,
        diffusion_time_embedding_dim=args.hidden_dim,
        n_times=args.n_timesteps
    ).to(device)
    
    protein_diffusion = ProteinDiffusion(
        model, 
        n_times=args.n_timesteps,
        beta_minmax=[1e-4, 2e-2],
        device=device
    ).to(device)
    
    optimizer = torch.optim.Adam(protein_diffusion.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()
    
    # Print model summary
    total_params = sum(p.numel() for p in protein_diffusion.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # Train the model
    protein_diffusion, losses = train_protein_diffusion(
        protein_diffusion,
        dataloader,
        optimizer,
        mse_loss,
        args.epochs,
        device
    )