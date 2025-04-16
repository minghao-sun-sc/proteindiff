import py3Dmol
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

def visualize_protein(pdb_file):
    """Visualize protein structure using py3Dmol"""
    with open(pdb_file) as f:
        pdb_data = f.read()
    
    view = py3Dmol.view(width=800, height=400)
    view.addModel(pdb_data, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    view.show()
    return view

def plot_training_progress(losses, title="Training Loss"):
    """Plot training loss over epochs"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def compare_structures(pred_pdb, true_pdb, title="Predicted vs. True Structure"):
    """Compare predicted and true protein structures"""
    with open(pred_pdb) as f:
        pred_data = f.read()
    with open(true_pdb) as f:
        true_data = f.read()
    
    view = py3Dmol.view(width=800, height=400)
    
    # Add predicted structure in blue
    view.addModel(pred_data, 'pdb')
    view.setStyle({'model': -1}, {'cartoon': {'color': 'blue'}})
    
    # Add true structure in green
    view.addModel(true_data, 'pdb')
    view.setStyle({'model': -1}, {'cartoon': {'color': 'green'}})
    
    view.zoomTo()
    view.show()
    return view