import os
import torch
import numpy as np
import argparse
import sys
import py3Dmol
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import time
from pathlib import Path
import tempfile
from Bio.PDB import PDBParser, PDBIO, Select, Structure, Model, Chain, Residue, Atom

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import one_hot_encode_sequence, save_protein_structure
from src.model import ProteinDenoiser
from src.diffusion import ProteinDiffusion

# Define amino acid properties for visualization
AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'size': 'small', 'polarity': 'nonpolar'},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'size': 'large', 'polarity': 'polar'},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'size': 'medium', 'polarity': 'polar'},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'size': 'medium', 'polarity': 'polar'},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'size': 'medium', 'polarity': 'nonpolar'},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'size': 'medium', 'polarity': 'polar'},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'size': 'medium', 'polarity': 'polar'},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'size': 'small', 'polarity': 'nonpolar'},
    'H': {'hydrophobicity': -3.2, 'charge': 0, 'size': 'medium', 'polarity': 'polar'},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'size': 'large', 'polarity': 'nonpolar'},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'size': 'large', 'polarity': 'nonpolar'},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'size': 'large', 'polarity': 'polar'},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'size': 'large', 'polarity': 'nonpolar'},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'size': 'large', 'polarity': 'nonpolar'},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'size': 'medium', 'polarity': 'nonpolar'},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'size': 'small', 'polarity': 'polar'},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'size': 'medium', 'polarity': 'polar'},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'size': 'large', 'polarity': 'nonpolar'},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'size': 'large', 'polarity': 'polar'},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'size': 'medium', 'polarity': 'nonpolar'}
}

def save_protein_with_sidechain_approximation(coords, sequence, output_file):
    """
    Save generated backbone coordinates as a PDB file with approximate side chain atoms.
    This enhances visualization by providing better volume for residues.
    """
    # Convert to numpy
    coords = coords.detach().cpu().numpy()
    
    # Create a simple PDB file
    with open(output_file, 'w') as f:
        atom_index = 1
        for i, residue_coords in enumerate(coords):
            if i >= len(sequence):
                break
            
            aa = sequence[i]
            
            # Write backbone atoms
            for j, atom_name in enumerate(["N", "CA", "C", "O"]):
                if j < len(residue_coords):
                    x, y, z = residue_coords[j]
                    f.write(f"ATOM  {atom_index:5d}  {atom_name:<3s} {aa:3s} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]}\n")
                    atom_index += 1
            
            # Add an approximate CB (beta carbon) atom based on the backbone
            if aa != 'G':  # Glycine doesn't have a CB
                if len(residue_coords) >= 3:
                    # Approximate CB position based on N, CA, C
                    n_pos = residue_coords[0]
                    ca_pos = residue_coords[1]
                    c_pos = residue_coords[2]
                    
                    # Vector from CA to midpoint of N and C
                    mid = (n_pos + c_pos) / 2
                    direction = ca_pos - mid
                    
                    # Normalize and scale for CB position
                    direction = direction / np.linalg.norm(direction) * 1.5
                    cb_pos = ca_pos + direction
                    
                    f.write(f"ATOM  {atom_index:5d}  {'CB':<3s} {aa:3s} A{i+1:4d}    {cb_pos[0]:8.3f}{cb_pos[1]:8.3f}{cb_pos[2]:8.3f}  1.00  0.00           C\n")
                    atom_index += 1
        
        f.write("END\n")

# def predict_secondary_structure(coords, sequence):
#     """
#     Simple secondary structure prediction based on backbone geometry.
#     Returns a list of secondary structure types ('H' for helix, 'E' for strand, 'C' for coil).
#     """
#     # Extract CA coordinates
#     ca_coords = coords[:, 1].detach().cpu().numpy()
    
#     # Initialize all as coil
#     ss_list = ['C'] * len(sequence)
    
#     # Check for helical patterns based on CA-CA-CA angles and distances
#     for i in range(1, len(ca_coords) - 1):
#         if i+3 < len(ca_coords):
#             # Check CA-CA distances for i, i+3 (characteristic of alpha helix)
#             dist = np.linalg.norm(ca_coords[i] - ca_coords[i+3])
#             if 4.5 < dist < 6.5:  # Typical alpha-helix i to i+3 distance
#                 ss_list[i] = 'H'
#                 ss_list[i+1] = 'H'
#                 ss_list[i+2] = 'H'
#                 ss_list[i+3] = 'H'
    
#     # Simple check for extended conformations (beta strands)
#     for i in range(1, len(ca_coords) - 2):
#         if i+2 < len(ca_coords):
#             v1 = ca_coords[i+1] - ca_coords[i]
#             v2 = ca_coords[i+2] - ca_coords[i+1]
#             # Check if vectors are roughly parallel (characteristic of beta strand)
#             cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#             if cos_angle > 0.8 and ss_list[i] == 'C':  # High cos means vectors point in similar direction
#                 ss_list[i] = 'E'
#                 ss_list[i+1] = 'E'
#                 ss_list[i+2] = 'E'
    
#     return ss_list

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

def visualize_enhanced_structure(coords, sequence, title=None, width=800, height=500, 
                                 view_style='cartoon', color_scheme='ss', show_surface=False, 
                                 spin=False, show_labels=False, show_sidechains=True):
    """
    Enhanced visualization of protein structure with multiple display options
    
    Args:
        coords: Protein coordinates tensor
        sequence: Amino acid sequence
        title: Title for the visualization
        width, height: Size of the viewer
        view_style: Main visualization style ('cartoon', 'stick', 'sphere', 'line')
        color_scheme: Coloring method ('ss', 'rainbow', 'chain', 'residue', 'hydrophobicity', 'charge')
        show_surface: Whether to display molecular surface
        spin: Whether to animate with spinning
        show_labels: Whether to show residue labels
        show_sidechains: Whether to approximate sidechains for better visualization
    """
    # Create a temporary PDB file
    temp_dir = Path("temp_pdb")
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / f"temp_{int(time.time() * 1000)}.pdb"
    
    # Save structure with or without sidechain approximation
    if show_sidechains:
        save_protein_with_sidechain_approximation(coords, sequence, temp_file)
    else:
        save_protein_structure(coords, sequence, temp_file)
    
    # Read the PDB file
    with open(temp_file) as f:
        pdb_data = f.read()
    
    # Create py3Dmol view
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_data, 'pdb')
    
    # Predict secondary structure for coloring
    ss_list = predict_secondary_structure(coords, sequence)
    
    # Set color scheme based on user selection
    if color_scheme == 'ss':
        # Color by secondary structure
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        for i, ss in enumerate(ss_list):
            sel = {'resi': i+1}
            if ss == 'H':
                view.setStyle(sel, {'cartoon': {'color': 'red'}})
            elif ss == 'E':
                view.setStyle(sel, {'cartoon': {'color': 'yellow'}})
            else:
                view.setStyle(sel, {'cartoon': {'color': 'green'}})
    
    elif color_scheme == 'rainbow':
        view.setStyle({'cartoon': {'color': 'spectrum'}})
    
    elif color_scheme == 'chain':
        view.setStyle({'cartoon': {'color': 'blue'}})
    
    elif color_scheme == 'residue':
        # Color by residue type
        for i, aa in enumerate(sequence):
            sel = {'resi': i+1}
            # Assign colors based on residue type
            if aa in 'AILMFWYV':  # Hydrophobic
                view.setStyle(sel, {'cartoon': {'color': 'yellow'}})
            elif aa in 'RKHED':  # Charged
                view.setStyle(sel, {'cartoon': {'color': 'red'}})
            elif aa in 'STNQC':  # Polar
                view.setStyle(sel, {'cartoon': {'color': 'green'}})
            else:  # Other
                view.setStyle(sel, {'cartoon': {'color': 'grey'}})
    
    elif color_scheme == 'hydrophobicity':
        # Color by hydrophobicity
        for i, aa in enumerate(sequence):
            if aa in AA_PROPERTIES:
                sel = {'resi': i+1}
                h_value = AA_PROPERTIES[aa]['hydrophobicity']
                # Create color gradient from blue (hydrophilic) to red (hydrophobic)
                if h_value <= -3.0:
                    color = 'blue'
                elif h_value <= 0:
                    color = 'cyan'
                elif h_value <= 2.0:
                    color = 'green'
                else:
                    color = 'red'
                view.setStyle(sel, {'cartoon': {'color': color}})
    
    elif color_scheme == 'charge':
        # Color by charge
        for i, aa in enumerate(sequence):
            if aa in AA_PROPERTIES:
                sel = {'resi': i+1}
                charge = AA_PROPERTIES[aa]['charge']
                if charge > 0:
                    color = 'blue'
                elif charge < 0:
                    color = 'red'
                else:
                    color = 'white'
                view.setStyle(sel, {'cartoon': {'color': color}})
    
    # Set the main visualization style
    if view_style == 'cartoon':
        view.setStyle({'cartoon': {}})
    elif view_style == 'stick':
        view.setStyle({'stick': {}})
    elif view_style == 'sphere':
        view.setStyle({'sphere': {}})
    elif view_style == 'line':
        view.setStyle({'line': {}})
    
    # Add surface if requested
    if show_surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'})
    
    # Add labels if requested
    if show_labels:
        for i, aa in enumerate(sequence):
            sel = {'resi': i+1}
            view.addLabel(f"{aa}{i+1}", {'position': 'CA', 'backgroundColor': 'white', 'fontColor': 'black'}, sel)
    
    # Set spin if requested
    if spin:
        view.spin(True)
    
    # Zoom to fit
    view.zoomTo()
    
    # Add title if provided
    if title:
        view.addLabel(title, {'position': {'x': 0, 'y': 0, 'z': -20}, 'backgroundColor': 'white', 'fontColor': 'black'})
    
    return view

def visualize_structure_comparison(predicted_coords, reference_pdb=None, sequence=None, 
                                  width=800, height=500, show_rmsd=True, color_scheme='chain'):
    """
    Compare predicted structure with reference structure side by side
    
    Args:
        predicted_coords: Predicted protein coordinates tensor
        reference_pdb: Path to reference PDB file (if available)
        sequence: Amino acid sequence
        width, height: Size of the viewer
        show_rmsd: Whether to calculate and display RMSD
        color_scheme: How to color the structures
    """
    # Create a temporary PDB file for predicted structure
    temp_dir = Path("temp_pdb")
    temp_dir.mkdir(exist_ok=True)
    predicted_file = temp_dir / f"predicted_{int(time.time() * 1000)}.pdb"
    
    # If sequence not provided but reference is, try to extract sequence from reference
    if sequence is None and reference_pdb is not None:
        try:
            parser = PDBParser(QUIET=True)
            ref_structure = parser.get_structure("reference", reference_pdb)
            sequence = ""
            for residue in ref_structure.get_residues():
                if residue.get_id()[0] == " ":  # Standard amino acid
                    from Bio.PDB.Polypeptide import three_to_one
                    try:
                        aa = three_to_one(residue.get_resname())
                        sequence += aa
                    except:
                        sequence += "X"
        except Exception as e:
            print(f"Could not extract sequence from reference: {e}")
            sequence = "A" * predicted_coords.shape[0]  # Default sequence
    
    # Save predicted structure
    save_protein_with_sidechain_approximation(predicted_coords, sequence, predicted_file)
    
    # Create py3Dmol view with two side-by-side viewers
    view = py3Dmol.view(linked=True, viewergrid=(1,2), width=width, height=height)
    
    # Add predicted structure to left viewer
    with open(predicted_file) as f:
        pred_data = f.read()
    view.addModel(pred_data, 'pdb', {'viewer': 0})
    
    # Set style for predicted structure
    if color_scheme == 'chain':
        view.setStyle({'cartoon': {'color': 'blue'}}, {'viewer': 0})
    else:
        view.setStyle({'cartoon': {'color': 'spectrum'}}, {'viewer': 0})
    view.addLabel("Predicted Structure", {'position': {'x': 0, 'y': 0, 'z': -20}, 
                                        'backgroundColor': 'white', 'fontColor': 'black'}, 
                 {'viewer': 0})
    
    # If reference structure is provided
    if reference_pdb and os.path.exists(reference_pdb):
        # Add reference structure to right viewer
        with open(reference_pdb) as f:
            ref_data = f.read()
        view.addModel(ref_data, 'pdb', {'viewer': 1})
        
        # Set style for reference structure
        if color_scheme == 'chain':
            view.setStyle({'cartoon': {'color': 'green'}}, {'viewer': 1})
        else:
            view.setStyle({'cartoon': {'color': 'spectrum'}}, {'viewer': 1})
        view.addLabel("Reference Structure", {'position': {'x': 0, 'y': 0, 'z': -20}, 
                                           'backgroundColor': 'white', 'fontColor': 'black'}, 
                     {'viewer': 1})
        
        # Calculate RMSD if requested
        if show_rmsd:
            try:
                # Parse both structures
                parser = PDBParser(QUIET=True)
                pred_struct = parser.get_structure("predicted", predicted_file)
                ref_struct = parser.get_structure("reference", reference_pdb)
                
                # Extract CA atoms
                pred_ca_atoms = [atom for atom in pred_struct.get_atoms() if atom.get_id() == 'CA']
                ref_ca_atoms = [atom for atom in ref_struct.get_atoms() if atom.get_id() == 'CA']
                
                # Use minimum number of atoms for comparison
                min_atoms = min(len(pred_ca_atoms), len(ref_ca_atoms))
                
                if min_atoms > 0:
                    # Calculate RMSD
                    pred_coords = np.array([atom.get_coord() for atom in pred_ca_atoms[:min_atoms]])
                    ref_coords = np.array([atom.get_coord() for atom in ref_ca_atoms[:min_atoms]])
                    
                    # Simple RMSD calculation (without superposition)
                    rmsd = np.sqrt(np.mean(np.sum((pred_coords - ref_coords)**2, axis=1)))
                    
                    # Add RMSD label
                    view.addLabel(f"RMSD: {rmsd:.2f} Å", {'position': {'x': 0, 'y': 5, 'z': -20}, 
                                                       'backgroundColor': 'white', 'fontColor': 'black'}, 
                                 {'viewer': 0})
            except Exception as e:
                print(f"Could not calculate RMSD: {e}")
    else:
        # If no reference, just add a message
        view.addModel("", 'pdb', {'viewer': 1})
        view.addLabel("No Reference Structure", {'position': {'x': 0, 'y': 0, 'z': 0}, 
                                               'backgroundColor': 'white', 'fontColor': 'black'}, 
                     {'viewer': 1})
    
    # Zoom both viewers to fit
    view.zoomTo()
    
    return view

def create_advanced_denoising_animation(structures, sequence, output_file, fps=5, mode='cartoon'):
    """
    Create a high-quality animation of the denoising process
    
    Args:
        structures: dictionary of structures at different timesteps
        sequence: amino acid sequence
        output_file: where to save the animation
        fps: frames per second
        mode: visualization mode ('cartoon', 'trace', 'both')
    """
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import LinearSegmentedColormap
    
    # Sort timesteps
    timesteps = sorted(structures.keys(), reverse=True)
    
    # Set up the figure
    if mode == 'both':
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        axes = [ax1, ax2]
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        axes = [ax]
    
    # Get backbone atoms for plotting
    backbone_coords = {}
    for t, struct in structures.items():
        # Get alpha carbon (CA) and other backbone atoms
        ca_coords = struct[:, 1, :].numpy()  # CA atoms
        c_coords = struct[:, 2, :].numpy()   # C atoms
        n_coords = struct[:, 0, :].numpy()   # N atoms
        backbone_coords[t] = (ca_coords, c_coords, n_coords)
    
    # Find global min and max for consistent scaling
    all_coords = []
    for ca, c, n in backbone_coords.values():
        all_coords.extend([ca, c, n])
    all_coords = np.concatenate(all_coords)
    x_min, y_min, z_min = all_coords.min(axis=0) - 2
    x_max, y_max, z_max = all_coords.max(axis=0) + 2
    
    # Create custom colormaps for secondary structure
    ss_colors = {'H': 'red', 'E': 'yellow', 'C': 'green'}
    
    # Predict secondary structures for each timestep
    ss_predictions = {}
    for t, struct in structures.items():
        ss_predictions[t] = predict_secondary_structure(struct, sequence)
    
    def init():
        for ax in axes:
            ax.clear()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        return []
    
    def update(frame):
        t = timesteps[frame]
        ca_coords, c_coords, n_coords = backbone_coords[t]
        ss_pred = ss_predictions[t]
        
        for i, ax in enumerate(axes):
            ax.clear()
            
            # Different visualization modes
            if mode == 'both':
                if i == 0:  # Cartoon representation
                    # Plot segments with colors based on predicted secondary structure
                    for j in range(len(ca_coords) - 1):
                        # Add safety check for ss_pred index
                        color = ss_colors.get(ss_pred[j] if j < len(ss_pred) else 'C', 'green')
                        ax.plot([ca_coords[j, 0], ca_coords[j+1, 0]], 
                                [ca_coords[j, 1], ca_coords[j+1, 1]], 
                                [ca_coords[j, 2], ca_coords[j+1, 2]], 
                                '-', color=color, linewidth=2)
                else:  # Trace representation with spheres
                    # Plot CA atoms as spheres
                    ax.scatter(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2], 
                              c=range(len(ca_coords)), cmap='viridis', s=50)
                    # Connect CA atoms with lines
                    ax.plot(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2], '-', color='blue', alpha=0.5)
            
            elif mode == 'cartoon':
                # Plot segments with colors based on predicted secondary structure
                for j in range(len(ca_coords) - 1):
                    # Add safety check for ss_pred index
                    color = ss_colors.get(ss_pred[j] if j < len(ss_pred) else 'C', 'green')
                    ax.plot([ca_coords[j, 0], ca_coords[j+1, 0]], 
                            [ca_coords[j, 1], ca_coords[j+1, 1]], 
                            [ca_coords[j, 2], ca_coords[j+1, 2]], 
                            '-', color=color, linewidth=2)
                    
                    # Add simplified visualization of secondary structure
                    # Add safety check for ss_pred index
                    if j < len(ss_pred) and ss_pred[j] == 'H':  # Helix
                        # Draw a thicker line for helices
                        ax.plot([ca_coords[j, 0], ca_coords[j+1, 0]], 
                                [ca_coords[j, 1], ca_coords[j+1, 1]], 
                                [ca_coords[j, 2], ca_coords[j+1, 2]], 
                                '-', color='red', linewidth=4)
                    elif j < len(ss_pred) and ss_pred[j] == 'E':  # Strand
                        # Add arrow-like appearance for strands
                        ax.quiver(ca_coords[j, 0], ca_coords[j, 1], ca_coords[j, 2],
                                 ca_coords[j+1, 0] - ca_coords[j, 0],
                                 ca_coords[j+1, 1] - ca_coords[j, 1],
                                 ca_coords[j+1, 2] - ca_coords[j, 2],
                                 color='yellow', arrow_length_ratio=0.5)
            
            elif mode == 'trace':
                # Plot CA atoms as spheres
                ax.scatter(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2], 
                          c=range(len(ca_coords)), cmap='viridis', s=50)
                # Connect CA atoms with lines
                ax.plot(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2], '-', color='blue', alpha=0.5)
            
            # Set axis limits and title
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            
            progress_percent = (max(timesteps) - t) / max(timesteps) * 100
            if mode == 'both':
                title = f"Timestep {t}" if i == 0 else f"Progress: {progress_percent:.1f}%"
            else:
                title = f"Protein Diffusion - Timestep {t} (Progress: {progress_percent:.1f}%)"
            ax.set_title(title)
            
            # Set viewing angle
            ax.view_init(elev=30, azim=frame % 360)
        
        return []
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(timesteps), init_func=init, blit=True, interval=1000/fps
    )
    
    # Save the animation
    anim.save(output_file, writer='pillow', fps=fps, dpi=150)
    print(f"Animation saved to {output_file}")
    
    plt.close(fig)
    return output_file

def visualize_enhanced_denoising_process(protein_diffusion, sequence, device, 
                                        timesteps_to_save=None, output_dir="diffusion_process",
                                        visualization_style='cartoon', color_scheme='ss',
                                        show_surface=False, reference_pdb=None):
    """
    Enhanced visualization of the protein structure denoising process
    
    Args:
        protein_diffusion: trained diffusion model
        sequence: amino acid sequence
        device: device to run the model on
        timesteps_to_save: list of timesteps to save (if None, sensible defaults will be used)
        output_dir: directory to save visualizations
        visualization_style: style for protein visualization
        color_scheme: how to color the structure
        show_surface: whether to display molecular surface
        reference_pdb: reference PDB file for comparison (if available)
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
            save_protein_with_sidechain_approximation(coords, sequence, output_file)
            pdb_files.append(output_file)
            
            # Create title
            progress_percent = (protein_diffusion.n_times - 1 - t) / (protein_diffusion.n_times - 1) * 100
            title = f"Timestep: {t} (Progress: {progress_percent:.1f}%)"
            
            # Create visualization
            vis = visualize_enhanced_structure(
                coords, 
                sequence, 
                title=title,
                view_style=visualization_style,
                color_scheme=color_scheme,
                show_surface=show_surface
            )
            visualizations.append((t, vis))
        
        # Create the final structure comparison if reference is provided
        if reference_pdb and os.path.exists(reference_pdb):
            final_coords = structures[0]  # The fully denoised structure
            comparison_vis = visualize_structure_comparison(
                final_coords,
                reference_pdb=reference_pdb,
                sequence=sequence,
                show_rmsd=True
            )
            visualizations.append(('comparison', comparison_vis))
        
        # Create animation
        animation_file = os.path.join(output_dir, "diffusion_animation.gif")
        create_advanced_denoising_animation(structures, sequence, animation_file, mode='both')
        
        # Create another animation with different style
        cartoon_animation_file = os.path.join(output_dir, "diffusion_cartoon_animation.gif")
        create_advanced_denoising_animation(structures, sequence, cartoon_animation_file, mode='cartoon')
        
        # Display visualizations in a Jupyter notebook
        try:
            from IPython.display import display, HTML
            
            print("\nShowing denoising process visualizations:")
            for t, vis in visualizations:
                if t == 'comparison':
                    print("\nComparison with reference structure:")
                else:
                    progress_percent = (protein_diffusion.n_times - 1 - t) / (protein_diffusion.n_times - 1) * 100
                    print(f"\nTimestep: {t} (Progress: {progress_percent:.1f}%)")
                display(vis.show())
            
            # Display animations
            print("\nDenoising process animation:")
            display(HTML(f'<img src="{animation_file}" alt="Diffusion Animation">'))
        except ImportError:
            print("\nTo view the visualizations interactively, run this script in a Jupyter notebook.")
            print(f"PDB files saved to {output_dir}")
            print(f"Animation saved to {animation_file}")
        
        return structures, pdb_files, visualizations, animation_file

def generate_multi_view_visualization(coords, sequence, output_file=None, 
                                     width=800, height=600, dpi=150, show_labels=False):
    """
    Generate a multi-view visualization of the protein structure showing different 
    representations and coloring schemes simultaneously
    
    Args:
        coords: Protein coordinates tensor
        sequence: Amino acid sequence
        output_file: File to save the visualization (if None, just returns the figure)
        width, height: Size of the figure in pixels
        dpi: DPI for the saved figure
        show_labels: Whether to show residue labels
    """
    # Create a figure with multiple subplots for different views
    fig, axs = plt.subplots(2, 3, figsize=(width/dpi, height/dpi), dpi=dpi)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Create a temporary PDB file for visualization
    temp_dir = Path("temp_pdb")
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / f"temp_multi_{int(time.time() * 1000)}.pdb"
    save_protein_with_sidechain_approximation(coords, sequence, temp_file)
    
    # Create py3Dmol views for each visualization style
    views = []
    titles = [
        "Cartoon (Rainbow)", "Surface", "Stick Model",
        "Secondary Structure", "Hydrophobicity", "Charge"
    ]
    
    # Visualization styles and options
    view_styles = ['cartoon', 'cartoon', 'stick', 'cartoon', 'cartoon', 'cartoon']
    color_schemes = ['rainbow', 'rainbow', 'element', 'ss', 'hydrophobicity', 'charge']
    show_surfaces = [False, True, False, False, False, False]
    
    # Create each visualization
    vis_data = []
    for i, (style, color, surface, title) in enumerate(zip(view_styles, color_schemes, show_surfaces, titles)):
        # Create visualization
        vis = visualize_enhanced_structure(
            coords, 
            sequence, 
            title=title,
            width=width//3,
            height=height//2,
            view_style=style,
            color_scheme=color,
            show_surface=surface,
            show_labels=(i==2 and show_labels)  # Only show labels on the stick model
        )
        vis_data.append(vis.write())
    
    # Plot each visualization in its subplot
    for i, (ax, data, title) in enumerate(zip(axs.flat, vis_data, titles)):
        row, col = i // 3, i % 3
        
        # Create HTML with embedded py3Dmol view
        html = f'<script src="https://3Dmol.org/build/3Dmol-min.js"></script>{data}'
        
        # Use matplotlib to display HTML content
        ax.axis('off')
        ax.text(0.5, 0.95, title, horizontalalignment='center', 
                verticalalignment='top', transform=ax.transAxes, fontsize=12)
        ax.text(0.5, 0.5, 'Interactive 3D View\n(Available in Jupyter)', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, style='italic', color='gray')
    
    # Add a main title
    plt.suptitle(f"Multi-view Visualization of Protein Structure ({len(sequence)} residues)", fontsize=14)
    
    # Save figure if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Multi-view visualization saved to {output_file}")
    
    return fig, vis_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced visualization of protein diffusion process')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--sequence', type=str, required=True, help='Amino acid sequence')
    parser.add_argument('--output_dir', type=str, default='enhanced_diffusion_vis', help='Output directory for visualizations')
    parser.add_argument('--n_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--timesteps', type=int, nargs='+', help='Specific timesteps to visualize')
    parser.add_argument('--style', type=str, default='cartoon', choices=['cartoon', 'stick', 'sphere', 'line'], 
                        help='Visualization style')
    parser.add_argument('--color', type=str, default='ss', 
                        choices=['ss', 'rainbow', 'chain', 'residue', 'hydrophobicity', 'charge'], 
                        help='Coloring scheme')
    parser.add_argument('--surface', action='store_true', help='Show molecular surface')
    parser.add_argument('--reference', type=str, help='Reference PDB file for comparison')
    parser.add_argument('--multi_view', action='store_true', help='Generate multi-view visualization of final structure')
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
    
    # Visualize the diffusion process with enhanced visualization
    structures, pdb_files, visualizations, animation_file = visualize_enhanced_denoising_process(
        protein_diffusion,
        args.sequence,
        device,
        timesteps_to_save=args.timesteps,
        output_dir=args.output_dir,
        visualization_style=args.style,
        color_scheme=args.color,
        show_surface=args.surface,
        reference_pdb=args.reference
    )
    
    # Generate multi-view visualization of final structure if requested
    if args.multi_view:
        final_coords = structures[0]  # The fully denoised structure
        multi_view_file = os.path.join(args.output_dir, "multi_view_visualization.png")
        generate_multi_view_visualization(
            final_coords,
            args.sequence,
            output_file=multi_view_file
        )