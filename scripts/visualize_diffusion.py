import os
import torch
import numpy as np
import argparse
import sys
import py3Dmol
from IPython.display import display

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import time
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import one_hot_encode_sequence, save_protein_structure
from src.model import ProteinDenoiser
from src.diffusion import ProteinDiffusion

def visualize_structure(coords, sequence, title=None, width=400, height=400):
    """Visualize a protein structure from coordinates"""
    # Create a temporary PDB file
    temp_dir = Path("temp_pdb")
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / f"temp_{int(time.time() * 1000)}.pdb"
    
    save_protein_structure(coords, sequence, temp_file)
    
    # Read the PDB file
    with open(temp_file) as f:
        pdb_data = f.read()
    
    # Create py3Dmol view
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_data, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    
    # Add title if provided
    if title:
        view.addLabel(title, {'position': {'x': 0, 'y': 0, 'z': 0}, 'backgroundColor': 'white', 'fontColor': 'black'})
    
    return view

def visualize_denoising_process(protein_diffusion, sequence, device, timesteps_to_save=None, output_dir="diffusion_process"):
    """
    Visualize the protein structure denoising process at different timesteps
    
    Args:
        protein_diffusion: trained diffusion model
        sequence: amino acid sequence
        device: device to run the model on
        timesteps_to_save: list of timesteps to save (if None, sensible defaults will be used)
        output_dir: directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a sensible default for timesteps to visualize if not provided
    if timesteps_to_save is None:
        # For a 1000-step process, visualize these steps
        total_steps = protein_diffusion.n_times
        if total_steps >= 1000:
            timesteps_to_save = [
                1000, 950, 900, 850, 800, 700, 600, 500, 400, 300, 200, 100, 50, 25, 10, 5, 1, 0
            ]
        else:
            # For fewer steps, calculate appropriate intervals
            num_points = min(18, total_steps)
            timesteps_to_save = sorted([
                int(i * total_steps / (num_points - 1)) for i in range(num_points)
            ], reverse=True)
            # Always include the first and last steps
            if 0 not in timesteps_to_save:
                timesteps_to_save.append(0)
            if total_steps - 1 not in timesteps_to_save:
                timesteps_to_save.append(total_steps - 1)
            timesteps_to_save = sorted(timesteps_to_save, reverse=True)
    
    # Ensure time step 0 (final result) is included
    if 0 not in timesteps_to_save:
        timesteps_to_save.append(0)
    
    print(f"Will visualize these timesteps: {timesteps_to_save}")
    
    protein_diffusion.eval()
    with torch.no_grad():
        # Convert sequence to one-hot encoding
        seq_encoding = one_hot_encode_sequence(sequence)
        seq_tensor = seq_encoding.unsqueeze(0).to(device)  # Add batch dimension
        
        # Start from random noise
        batch_size, seq_len, num_aa = seq_tensor.shape
        x_t = torch.randn((batch_size, seq_len, 4, 3)).to(device)  # 4 backbone atoms per residue
        
        # Save initial noise
        initial_noise = x_t.clone()
        
        # Dictionary to store structures at different timesteps
        structures = {}
        structures[protein_diffusion.n_times - 1] = initial_noise[0].cpu()  # Store initial noise
        
        # Iteratively denoise
        for t in tqdm(range(protein_diffusion.n_times-1, -1, -1), desc="Sampling"):
            timestep = torch.tensor([t]).repeat(batch_size).long().to(device)
            x_t = protein_diffusion.denoise_at_t(x_t, seq_tensor, timestep, t)
            
            # Save structure at specified timesteps
            if t in timesteps_to_save:
                structures[t] = x_t[0].cpu()  # Store the first batch item (we only have one)
        
        # Visualize and save all collected structures
        visualizations = []
        pdb_files = []
        
        for t, coords in structures.items():
            # Save as PDB file
            output_file = os.path.join(output_dir, f"structure_timestep_{t}.pdb")
            save_protein_structure(coords, sequence, output_file)
            pdb_files.append(output_file)
            
            # Create title
            progress_percent = (protein_diffusion.n_times - 1 - t) / (protein_diffusion.n_times - 1) * 100
            title = f"Timestep: {t} (Progress: {progress_percent:.1f}%)"
            
            # Create visualization
            vis = visualize_structure(coords, sequence, title)
            visualizations.append((t, vis))
        
        # Display visualizations in a Jupyter notebook
        try:
            from IPython.display import display, HTML
            
            print("\nShowing denoising process visualizations:")
            for t, vis in visualizations:
                progress_percent = (protein_diffusion.n_times - 1 - t) / (protein_diffusion.n_times - 1) * 100
                print(f"\nTimestep: {t} (Progress: {progress_percent:.1f}%)")
                display(vis.show())
        except ImportError:
            print("\nTo view the visualizations interactively, run this script in a Jupyter notebook.")
            print(f"PDB files saved to {output_dir}")
        
        return structures, pdb_files, visualizations

def create_denoising_animation(structures, sequence, output_file, fps=5):
    """
    Create an animation of the denoising process
    
    Args:
        structures: dictionary of structures at different timesteps
        sequence: amino acid sequence
        output_file: where to save the animation
        fps: frames per second
    """
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    
    # Sort timesteps
    timesteps = sorted(structures.keys(), reverse=True)
    
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get CA atoms (alpha carbon) for plotting - index 1 in our coordinates
    ca_coords = {t: struct[:, 1, :].numpy() for t, struct in structures.items()}
    
    # Find global min and max for consistent scaling
    all_coords = np.concatenate([coords for coords in ca_coords.values()])
    x_min, y_min, z_min = all_coords.min(axis=0) - 1
    x_max, y_max, z_max = all_coords.max(axis=0) + 1
    
    def init():
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_title("Protein Diffusion Process")
        return []
    
    def update(frame):
        t = timesteps[frame]
        ax.clear()
        
        # Get CA coordinates for this timestep
        coords = ca_coords[t]
        
        # Plot the CA trace
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'o-', markersize=2, linewidth=1)
        
        # Connect CA atoms with lines, but only for sequential residues
        for i in range(len(coords) - 1):
            ax.plot([coords[i, 0], coords[i+1, 0]], 
                    [coords[i, 1], coords[i+1, 1]], 
                    [coords[i, 2], coords[i+1, 2]], '-', color='blue')
        
        # Set axis limits and title
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        progress_percent = (protein_diffusion.n_times - 1 - t) / (protein_diffusion.n_times - 1) * 100
        ax.set_title(f"Protein Diffusion - Timestep {t} (Progress: {progress_percent:.1f}%)")
        
        return []
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(timesteps), init_func=init, blit=True, interval=1000/fps
    )
    
    # Save the animation
    anim.save(output_file, writer='pillow', fps=fps)
    print(f"Animation saved to {output_file}")
    
    plt.close(fig)
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize protein diffusion process')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--sequence', type=str, required=True, help='Amino acid sequence')
    parser.add_argument('--output_dir', type=str, default='diffusion_process', help='Output directory for visualizations')
    parser.add_argument('--n_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--timesteps', type=int, nargs='+', help='Specific timesteps to visualize')
    parser.add_argument('--create_animation', action='store_true', help='Create animation of the diffusion process')
    args = parser.parse_args()
    
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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    protein_diffusion.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    
    # Visualize the diffusion process
    structures, pdb_files, visualizations = visualize_denoising_process(
        protein_diffusion,
        args.sequence,
        device,
        timesteps_to_save=args.timesteps,
        output_dir=args.output_dir
    )
    
    # Create animation if requested
    if args.create_animation:
        animation_file = os.path.join(args.output_dir, "diffusion_animation.gif")
        create_denoising_animation(structures, args.sequence, animation_file)