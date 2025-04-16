import os
import torch
import numpy as np
import argparse
import sys

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import one_hot_encode_sequence, save_protein_structure
from src.model import ProteinDenoiser
from src.diffusion import ProteinDiffusion
from src.visualization import visualize_protein

def generate_protein_structures(protein_diffusion, sequences, device, output_dir="generated_proteins"):
    """Generate protein structures for given sequences"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    protein_diffusion.eval()
    with torch.no_grad():
        # Convert sequences to one-hot encoding
        seq_encodings = [one_hot_encode_sequence(seq) for seq in sequences]
        seq_tensor = torch.stack(seq_encodings).to(device)
        
        # Generate coordinates
        print("Generating protein structures...")
        generated_coords = protein_diffusion.sample(seq_tensor)
        
        # Save structures as PDB files
        output_files = []
        for i, (coords, seq) in enumerate(zip(generated_coords, sequences)):
            output_file = os.path.join(output_dir, f"generated_protein_{i}.pdb")
            save_protein_structure(coords, seq, output_file)
            print(f"Saved generated structure to {output_file}")
            output_files.append(output_file)
        
        return generated_coords, sequences, output_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate protein structures with ProteinDiff')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--sequences', type=str, nargs='+', required=True, help='Amino acid sequences')
    parser.add_argument('--output_dir', type=str, default='generated_proteins', help='Output directory for generated structures')
    parser.add_argument('--n_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--visualize', action='store_true', help='Visualize the generated structures')
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
    
    # Generate structures
    _, _, output_files = generate_protein_structures(
        protein_diffusion,
        args.sequences,
        device,
        args.output_dir
    )
    
    # Visualize if requested
    if args.visualize:
        for i, pdb_file in enumerate(output_files):
            print(f"\nVisualizing structure for sequence {i+1}:")
            view = visualize_protein(pdb_file)