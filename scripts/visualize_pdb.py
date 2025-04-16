import os
import sys
import argparse
import py3Dmol
from IPython.display import display
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_pdb_structure(pdb_file, style='cartoon', color_scheme='spectrum', 
                           show_surface=False, width=800, height=500, 
                           spin=False, show_labels=False, save_image=None):
    """
    Visualize a protein structure from a PDB file using py3Dmol
    
    Args:
        pdb_file: Path to PDB file
        style: 'cartoon', 'stick', 'sphere', 'line'
        color_scheme: 'spectrum', 'chain', 'residue', 'element'
        show_surface: Whether to show molecular surface
        width, height: Viewport dimensions
        spin: Whether to animate with spinning
        show_labels: Whether to show residue labels
        save_image: Optional filename to save a static image
    """
    # Check if PDB file exists
    if not os.path.exists(pdb_file):
        print(f"Error: PDB file {pdb_file} not found.")
        return None
    
    # Read PDB file
    with open(pdb_file) as f:
        pdb_data = f.read()
    
    # Create py3Dmol view
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_data, 'pdb')
    
    # Set style based on user selection
    if style == 'cartoon':
        if color_scheme == 'spectrum':
            view.setStyle({'cartoon': {'color': 'spectrum'}})
        elif color_scheme == 'chain':
            view.setStyle({'cartoon': {'colorscheme': 'chain'}})
        elif color_scheme == 'residue':
            view.setStyle({'cartoon': {'colorscheme': 'amino'}})
        elif color_scheme == 'element':
            view.setStyle({'cartoon': {'color': 'element'}})
        elif color_scheme == 'secondary':
            # Color by predicted secondary structure
            view.setStyle({'cartoon': {}})
            view.setStyle({'cartoon': {'color':'green'}})
            view.setStyle({'helix': {'cartoon': {'color':'red'}}})
            view.setStyle({'sheet': {'cartoon': {'color':'yellow'}}})
    elif style == 'stick':
        view.setStyle({'stick': {'colorscheme': color_scheme}})
    elif style == 'sphere':
        view.setStyle({'sphere': {'colorscheme': color_scheme}})
    elif style == 'line':
        view.setStyle({'line': {'colorscheme': color_scheme}})
    
    # Add surface if requested
    if show_surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'colorscheme': 'whiteCarbon'})
    
    # Add labels if requested
    if show_labels:
        view.addResLabels()
    
    # Set spin if requested
    if spin:
        view.spin(True)
    
    # Zoom to fit
    view.zoomTo()
    
    # Save static image if requested
    if save_image:
        # Currently, py3Dmol doesn't support direct image saving,
        # but we can use matplotlib to capture the view
        if not save_image.endswith(('.png', '.jpg', '.jpeg')):
            save_image += '.png'
        
        # Create a figure to hold the image
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.5, "py3Dmol visualization\n(static screenshot not available)", 
               ha='center', va='center', fontsize=12)
        fig.savefig(save_image, bbox_inches='tight')
        plt.close(fig)
        print(f"Image representation saved to {save_image}")
        print("Note: This is a placeholder. For true 3D visualization, run this script in a Jupyter notebook.")
    
    return view

def main():
    parser = argparse.ArgumentParser(description='Visualize a PDB file using py3Dmol')
    parser.add_argument('pdb_file', type=str, help='Path to PDB file to visualize')
    parser.add_argument('--style', type=str, default='cartoon', 
                        choices=['cartoon', 'stick', 'sphere', 'line'], 
                        help='Visualization style')
    parser.add_argument('--color', type=str, default='spectrum', 
                        choices=['spectrum', 'chain', 'residue', 'element', 'secondary'], 
                        help='Color scheme')
    parser.add_argument('--surface', action='store_true', help='Show molecular surface')
    parser.add_argument('--width', type=int, default=800, help='Viewport width')
    parser.add_argument('--height', type=int, default=500, help='Viewport height')
    parser.add_argument('--spin', action='store_true', help='Animate with spinning')
    parser.add_argument('--labels', action='store_true', help='Show residue labels')
    parser.add_argument('--save', type=str, help='Save a static image')
    
    args = parser.parse_args()
    
    # Visualize PDB structure
    view = visualize_pdb_structure(
        args.pdb_file,
        style=args.style,
        color_scheme=args.color,
        show_surface=args.surface,
        width=args.width,
        height=args.height,
        spin=args.spin,
        show_labels=args.labels,
        save_image=args.save
    )
    
    # Check if running in a Jupyter notebook
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            # Display the visualization in the notebook
            display(view.show())
            print("Visualization displayed in notebook. If you don't see it, ensure py3Dmol is installed.")
        else:
            print("Interactive visualization is only available in Jupyter notebooks.")
            print("To view this visualization, run this script in a Jupyter notebook environment.")
            print(f"Alternatively, you can use other tools like PyMOL or VMD to view the PDB file: {args.pdb_file}")
    except ImportError:
        print("Interactive visualization is only available in Jupyter notebooks.")
        print("To view this visualization, run this script in a Jupyter notebook environment.")
        print(f"Alternatively, you can use other tools like PyMOL or VMD to view the PDB file: {args.pdb_file}")

if __name__ == "__main__":
    main()