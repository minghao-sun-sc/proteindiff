import os
import torch
import numpy as np
import argparse
import py3Dmol
from pathlib import Path
import imageio
import tempfile
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import sys

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load the necessary components from your model
from src.data_utils import one_hot_encode_sequence, save_protein_structure
from src.model import ProteinDenoiser
from src.diffusion import ProteinDiffusion

def predict_secondary_structure(coords, sequence):
    """
    Simple secondary structure prediction based on backbone geometry.
    Returns a list of secondary structure types ('H' for helix, 'E' for strand, 'C' for coil).
    """
    # Extract CA coordinates
    ca_coords = coords[:, 1].detach().cpu().numpy()
    
    # Initialize all as coil
    sequence_length = len(sequence)
    if sequence_length == 0:  # Guard against empty sequences
        return []
    
    ss_list = ['C'] * sequence_length
    
    # Check for helical patterns based on CA-CA-CA angles and distances
    for i in range(1, min(len(ca_coords) - 1, sequence_length)):
        if i+3 < min(len(ca_coords), sequence_length):
            # Check CA-CA distances for i, i+3 (characteristic of alpha helix)
            dist = np.linalg.norm(ca_coords[i] - ca_coords[i+3])
            if 4.5 < dist < 6.5:  # Typical alpha-helix i to i+3 distance
                ss_list[i] = 'H'
                ss_list[i+1] = 'H'
                ss_list[i+2] = 'H'
                ss_list[i+3] = 'H'
    
    # Simple check for extended conformations (beta strands)
    for i in range(1, min(len(ca_coords) - 2, sequence_length)):
        if i+2 < min(len(ca_coords), sequence_length):
            v1 = ca_coords[i+1] - ca_coords[i]
            v2 = ca_coords[i+2] - ca_coords[i+1]
            # Check if vectors are roughly parallel (characteristic of beta strand)
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            # Avoid division by zero
            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                # Make sure i is a valid index for ss_list
                if 0 <= i < len(ss_list) and cos_angle > 0.8 and ss_list[i] == 'C':
                    ss_list[i] = 'E'
                    if i+1 < len(ss_list):
                        ss_list[i+1] = 'E'
                    if i+2 < len(ss_list):
                        ss_list[i+2] = 'E'
    
    return ss_list

def create_3d_visualization(coords, sequence, ss_pred, output_file, view_angle=(30, 45), dpi=150):
    """
    Create a 3D visualization of the protein structure with secondary structure coloring.
    
    Args:
        coords: Coordinates of backbone atoms (batch_size, seq_len, num_atoms, 3)
        sequence: Amino acid sequence
        ss_pred: Predicted secondary structure ('H', 'E', 'C')
        output_file: Where to save the image
        view_angle: (elevation, azimuth) for the 3D view
        dpi: DPI for the output image
    """
    # Extract coordinates
    ca_coords = coords[:, 1].detach().cpu().numpy()  # CA atoms
    
    # Create figure
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for secondary structure
    ss_colors = {'H': 'red', 'E': 'yellow', 'C': 'green'}
    
    # Plot the backbone with secondary structure coloring
    for i in range(len(ca_coords) - 1):
        # Get color based on secondary structure
        if i < len(ss_pred):
            color = ss_colors.get(ss_pred[i], 'green')
        else:
            color = 'green'
            
        # Draw backbone segment
        ax.plot([ca_coords[i, 0], ca_coords[i+1, 0]],
                [ca_coords[i, 1], ca_coords[i+1, 1]],
                [ca_coords[i, 2], ca_coords[i+1, 2]],
                color=color, linewidth=2)
        
        # Add visual cues for secondary structure
        if i < len(ss_pred):
            if ss_pred[i] == 'H':  # Helix
                # Draw a thicker line for helices
                ax.plot([ca_coords[i, 0], ca_coords[i+1, 0]],
                        [ca_coords[i, 1], ca_coords[i+1, 1]],
                        [ca_coords[i, 2], ca_coords[i+1, 2]],
                        color='red', linewidth=4, alpha=0.7)
                
            elif ss_pred[i] == 'E':  # Beta strand
                # Add arrow-like appearance for strands
                if i+1 < len(ca_coords):
                    arrow_vector = ca_coords[i+1] - ca_coords[i]
                    ax.quiver(ca_coords[i, 0], ca_coords[i, 1], ca_coords[i, 2],
                              arrow_vector[0], arrow_vector[1], arrow_vector[2],
                              color='yellow', arrow_length_ratio=0.3, alpha=0.7)
    
    # Plot CA atoms
    ax.scatter(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2], 
               c=[ss_colors.get(ss, 'green') for ss in ss_pred[:len(ca_coords)]], 
               s=30, alpha=0.8)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label='Alpha Helix'),
        Line2D([0], [0], color='yellow', lw=4, label='Beta Strand'),
        Line2D([0], [0], color='green', lw=4, label='Coil/Loop')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Add sequence information
    seq_text = f"Sequence: {sequence[:10]}..." if len(sequence) > 10 else f"Sequence: {sequence}"
    ax.text2D(0.05, 0.95, seq_text, transform=ax.transAxes, fontsize=10)
    
    # Set axis labels
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    
    # Set title
    ax.set_title('Protein Structure with Secondary Structure Elements', fontsize=12)
    
    # Save the figure
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_file

def create_multi_view_visualization(coords, sequence, ss_pred, output_file, dpi=150):
    """Create a visualization with multiple views of the protein structure"""
    # Extract coordinates
    ca_coords = coords[:, 1].detach().cpu().numpy()  # CA atoms
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(16, 10), dpi=dpi)
    
    # Define viewing angles (elevation, azimuth)
    views = [(30, 0), (30, 90), (30, 180), (30, 270), (90, 0), (0, 0)]
    view_names = ["Front", "Side 1", "Back", "Side 2", "Top", "Bottom"]
    
    # Define colors for secondary structure
    ss_colors = {'H': 'red', 'E': 'yellow', 'C': 'green'}
    
    # Plot the structure from different angles
    for i, ((elev, azim), name) in enumerate(zip(views, view_names)):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        
        # Plot the backbone with secondary structure coloring
        for j in range(len(ca_coords) - 1):
            # Get color based on secondary structure
            if j < len(ss_pred):
                color = ss_colors.get(ss_pred[j], 'green')
            else:
                color = 'green'
                
            # Draw backbone segment
            ax.plot([ca_coords[j, 0], ca_coords[j+1, 0]],
                    [ca_coords[j, 1], ca_coords[j+1, 1]],
                    [ca_coords[j, 2], ca_coords[j+1, 2]],
                    color=color, linewidth=2)
            
            # Add visual cues for secondary structure
            if j < len(ss_pred):
                if ss_pred[j] == 'H' and j+1 < len(ca_coords):  # Helix
                    # Draw a thicker line for helices
                    ax.plot([ca_coords[j, 0], ca_coords[j+1, 0]],
                            [ca_coords[j, 1], ca_coords[j+1, 1]],
                            [ca_coords[j, 2], ca_coords[j+1, 2]],
                            color='red', linewidth=4, alpha=0.7)
                    
                elif ss_pred[j] == 'E' and j+1 < len(ca_coords):  # Beta strand
                    # Add arrow-like appearance for strands
                    arrow_vector = ca_coords[j+1] - ca_coords[j]
                    ax.quiver(ca_coords[j, 0], ca_coords[j, 1], ca_coords[j, 2],
                            arrow_vector[0], arrow_vector[1], arrow_vector[2],
                            color='yellow', arrow_length_ratio=0.3, alpha=0.7)
        
        # # Plot CA atoms
        # ax.scatter(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2], 
        #         c=[ss_colors.get(ss, 'green') for ss in ss_pred[:len(ca_coords)]], 
        #         s=20, alpha=0.8)
        # Ensure the color list matches the number of coordinates
        color_list = []
        for j in range(len(ca_coords)):
            if j < len(ss_pred):
                color_list.append(ss_colors.get(ss_pred[j], 'green'))
            else:
                color_list.append('green')

        ax.scatter(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2], 
                c=color_list, s=20, alpha=0.8)
        
        # Set title for the view
        ax.set_title(f"{name} View", fontsize=10)
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        
        # Remove axis labels to save space
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        # Add subtle grid for better depth perception
        ax.grid(True, alpha=0.3)
    
    # Add a title for the overall figure
    plt.suptitle(f"Multi-view Protein Structure Visualization\nSequence: {sequence[:20]}...", fontsize=14)
    
    # Add a single legend for the entire figure
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label='Alpha Helix'),
        Line2D([0], [0], color='yellow', lw=4, label='Beta Strand'),
        Line2D([0], [0], color='green', lw=4, label='Coil/Loop')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_file

def save_structure_visualization(structure, sequence, output_path, multi_view=False):
    """
    Save a visualization of the protein structure with secondary structure annotation
    
    Args:
        structure: Tensor of atom coordinates
        sequence: Amino acid sequence
        output_path: Where to save the visualization
        multi_view: Whether to create a multi-view visualization
    """
    # Predict secondary structure
    ss_pred = predict_secondary_structure(structure, sequence)
    
    # Create visualization
    if multi_view:
        return create_multi_view_visualization(structure, sequence, ss_pred, output_path)
    else:
        return create_3d_visualization(structure, sequence, ss_pred, output_path)

def visualize_denoising_process(protein_diffusion, sequence, device, 
                               output_dir="denoising_visualization",
                               num_steps_to_show=10,
                               multi_view=False):
    """
    Generate and save visualizations of the protein denoising process
    
    Args:
        protein_diffusion: The diffusion model
        sequence: Amino acid sequence
        device: Device to run on (cuda/cpu)
        output_dir: Directory to save visualizations
        num_steps_to_show: Number of timesteps to visualize
        multi_view: Whether to use multi-view visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temp directory for PDB files
    temp_dir = os.path.join(output_dir, "temp_pdb")
    os.makedirs(temp_dir, exist_ok=True)
    
    protein_diffusion.eval()
    with torch.no_grad():
        # Convert sequence to one-hot encoding
        seq_encoding = one_hot_encode_sequence(sequence)
        seq_tensor = seq_encoding.unsqueeze(0).to(device)  # Add batch dimension
        
        # Start from random noise
        batch_size, seq_len, num_aa = seq_tensor.shape
        x_t = torch.randn((batch_size, seq_len, 4, 3)).to(device)  # 4 backbone atoms per residue
        
        # Calculate timesteps to save
        total_steps = protein_diffusion.n_times
        step_indices = np.linspace(0, total_steps-1, num_steps_to_show, dtype=int)
        steps_to_save = sorted([total_steps - 1 - i for i in step_indices])
        print(f"Will visualize these timesteps: {steps_to_save}")
        
        # Dictionary to store structures at different timesteps
        structures = {}
        
        # List to store images for GIF
        frames = []
        pdb_files = []
        
        # Iteratively denoise
        for t in tqdm(range(total_steps-1, -1, -1), desc="Sampling"):
            timestep = torch.tensor([t]).repeat(batch_size).long().to(device)
            x_t = protein_diffusion.denoise_at_t(x_t, seq_tensor, timestep, t)
            
            # Save visualization at specified timesteps
            if t in steps_to_save:
                # Store the structure
                structures[t] = x_t[0].clone()
                
                # Save PDB file
                pdb_file = os.path.join(temp_dir, f"structure_timestep_{t}.pdb")
                save_protein_structure(x_t[0].cpu(), sequence, pdb_file)
                pdb_files.append(pdb_file)
                
                # Create and save visualization
                img_file = os.path.join(output_dir, f"structure_timestep_{t}.png")
                save_structure_visualization(x_t[0].clone(), sequence, img_file, multi_view)
                frames.append(img_file)
        
        # Create GIF animation
        gif_file = os.path.join(output_dir, "denoising_animation.gif")
        with imageio.get_writer(gif_file, mode='I', duration=0.5) as writer:
            for frame in frames:
                image = imageio.imread(frame)
                writer.append_data(image)
        
        print(f"Denoising animation saved to {gif_file}")
        
        # Create a simple 3D animation showing CA atom traces
        create_denoising_animation(structures, sequence, 
                                 os.path.join(output_dir, "denoising_trace.gif"))
        
        return pdb_files, frames, gif_file, structures

def create_denoising_animation(structures, sequence, output_file, fps=5):
    """Create an animation of the protein folding process with secondary structure"""
    # Sort timesteps
    timesteps = sorted(structures.keys(), reverse=True)
    
    # Set up the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for secondary structure
    ss_colors = {'H': 'red', 'E': 'yellow', 'C': 'green'}
    
    # Calculate secondary structure for each timestep
    ss_predictions = {}
    for t, struct in structures.items():
        ss_predictions[t] = predict_secondary_structure(struct, sequence)
    
    # Find global min and max for consistent scaling
    all_coords = []
    for struct in structures.values():
        all_coords.append(struct[:, 1, :].detach().cpu().numpy())  # CA atoms
    all_coords = np.concatenate(all_coords, axis=0)
    min_vals = np.min(all_coords, axis=0) - 2
    max_vals = np.max(all_coords, axis=0) + 2
    
    # Create frames
    frames = []
    for i, t in enumerate(timesteps):
        # Clear figure
        ax.clear()
        
        # Get structure and secondary structure
        struct = structures[t]
        ss_pred = ss_predictions[t]
        
        # Extract CA coordinates
        ca_coords = struct[:, 1, :].detach().cpu().numpy()
        
        # Plot the backbone with secondary structure coloring
        for j in range(len(ca_coords) - 1):
            # Get color based on secondary structure (with bounds checking)
            if j < len(ss_pred):
                color = ss_colors.get(ss_pred[j], 'green')
            else:
                color = 'green'
            
            # Draw backbone segment
            ax.plot([ca_coords[j, 0], ca_coords[j+1, 0]],
                    [ca_coords[j, 1], ca_coords[j+1, 1]],
                    [ca_coords[j, 2], ca_coords[j+1, 2]],
                    color=color, linewidth=2)
            
            # Add visual cues for secondary structure
            if j < len(ss_pred):
                if ss_pred[j] == 'H' and j+1 < len(ca_coords):  # Helix
                    # Draw a thicker line for helices
                    ax.plot([ca_coords[j, 0], ca_coords[j+1, 0]],
                            [ca_coords[j, 1], ca_coords[j+1, 1]],
                            [ca_coords[j, 2], ca_coords[j+1, 2]],
                            color='red', linewidth=4, alpha=0.7)
                    
                elif ss_pred[j] == 'E' and j+1 < len(ca_coords):  # Beta strand
                    # Add arrow-like appearance for strands
                    arrow_vector = ca_coords[j+1] - ca_coords[j]
                    ax.quiver(ca_coords[j, 0], ca_coords[j, 1], ca_coords[j, 2],
                            arrow_vector[0], arrow_vector[1], arrow_vector[2],
                            color='yellow', arrow_length_ratio=0.3, alpha=0.7)
        
        # # Plot CA atoms
        # ax.scatter(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2], 
        #         c=[ss_colors.get(ss, 'green') if i < len(ss_pred) else 'green' for i, ss in enumerate(ss_pred)], 
        #         s=30, alpha=0.8)

        # Ensure the color list matches the number of coordinates
        color_list = []
        for j in range(len(ca_coords)):
            if j < len(ss_pred):
                color_list.append(ss_colors.get(ss_pred[j], 'green'))
            else:
                color_list.append('green')

        ax.scatter(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2], 
                c=color_list, s=30, alpha=0.8)
        
        # Set axis limits for consistent view
        ax.set_xlim(min_vals[0], max_vals[0])
        ax.set_ylim(min_vals[1], max_vals[1])
        ax.set_zlim(min_vals[2], max_vals[2])
        
        # Set labels
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=4, label='Alpha Helix'),
            Line2D([0], [0], color='yellow', lw=4, label='Beta Strand'),
            Line2D([0], [0], color='green', lw=4, label='Coil/Loop')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set title
        progress_percent = (max(timesteps) - t) / max(timesteps) * 100
        ax.set_title(f"Timestep: {t} (Progress: {progress_percent:.1f}%)")
        
        # Set viewing angle with some rotation
        ax.view_init(elev=30, azim=(i * 10) % 360)
        
        # Save frame
        frame_file = f"temp_frame_{i:03d}.png"
        plt.savefig(frame_file, dpi=150, bbox_inches='tight')
        frames.append(frame_file)
    
    # Create GIF
    with imageio.get_writer(output_file, mode='I', duration=1000/fps) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)
    
    # Clean up temporary files
    for frame in frames:
        if os.path.exists(frame):
            os.remove(frame)
    
    print(f"Denoising animation saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Visualize protein diffusion process')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--sequence', type=str, required=True, help='Amino acid sequence')
    parser.add_argument('--output_dir', type=str, default='denoising_visualization', help='Output directory for visualizations')
    parser.add_argument('--n_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--num_steps_to_show', type=int, default=10, help='Number of timesteps to visualize')
    parser.add_argument('--multi_view', action='store_true', help='Use multi-view visualization')
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
    pdb_files, frames, gif_file, structures = visualize_denoising_process(
        protein_diffusion,
        args.sequence,
        device,
        args.output_dir,
        args.num_steps_to_show,
        args.multi_view
    )
    
    # Print information about output files
    print("\nVisualization results:")
    print(f"1. PDB files: {len(pdb_files)} structures saved in {os.path.join(args.output_dir, 'temp_pdb')}")
    print(f"2. Static images: {len(frames)} images saved in {args.output_dir}")
    print(f"3. Animation files: ")
    print(f"   - Denoising animation: {gif_file}")
    print(f"   - Trace animation: {os.path.join(args.output_dir, 'denoising_trace.gif')}")

if __name__ == "__main__":
    main()