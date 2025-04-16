# ProteinDiff: Protein Structure Prediction with Diffusion Models

This repository contains code for implementing a protein structure prediction model using diffusion models based on PyTorch. The approach uses a denoising diffusion probabilistic model to learn and generate protein backbone structures from amino acid sequences.

## Overview

ProteinDiff implements a diffusion-based generative model that:
1. Takes an amino acid sequence as input
2. Generates a plausible 3D protein backbone structure
3. Visualizes the structure with secondary structure elements (helices, sheets, coils)

The model progressively denoises random noise into protein coordinates, guided by the amino acid sequence information.

## Installation

### Requirements

Clone the repository and set up the environment:

```bash
git clone https://github.com/yourusername/proteindiff.git
cd proteindiff
```

### Environment Setup

Create a conda environment with all required dependencies:

```bash
conda create -n diffusion-tutor python=3.8
conda activate diffusion-tutor
pip install torch torchvision
pip install numpy matplotlib biopython tqdm py3Dmol imageio
```

Or use the provided environment file:

```bash
conda env create -f environment.yml
conda activate diffusion-tutor
```

## Project Structure

```
proteindiff/
├── data/
│   └── pdb_files/        # Folder for PDB training data
├── models/               # Saved model checkpoints
├── scripts/
│   ├── train.py          # Training script
│   ├── sample.py         # Sampling from trained model
│   └── visualize_denoising.py  # Visualization of denoising process
├── src/
│   ├── data_utils.py     # Data loading and preprocessing utilities
│   ├── diffusion.py      # Diffusion model implementation
│   └── model.py          # Neural network architecture
└── README.md
```

## Usage

### Training

To train the model on your protein dataset:

```bash
python scripts/train.py --data_dir data/pdb_files --epochs 50 --batch_size 8
```

This will train the model and save checkpoints in the `models/` directory.

### Sampling and Visualization

To generate a protein structure from an amino acid sequence:

```bash
python scripts/sample.py --checkpoint models/proteindiff_epoch_50.pt --sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGY" --output_pdb output/sampled_protein.pdb
```

To visualize the denoising process:

```bash
python scripts/visualize_denoising.py --checkpoint models/proteindiff_epoch_50.pt --sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGY" --multi_view
```

This will create a series of visualizations showing the denoising process from random noise to protein structure. The `--multi_view` flag generates visualizations from multiple angles.

### Output Files

The visualization script generates:
1. PDB files for each denoising step
2. Static images showing the 3D structure with secondary structure elements colored
3. Animation files showing the denoising process:
   - Full structure animation with secondary structure elements
   - CA trace animation showing protein backbone evolution

## Model Details

### Architecture

- **Backbone**: Transformer-based architecture for processing amino acid sequences and coordinates
- **Diffusion Process**: Forward and reverse diffusion processes with noise scheduling
- **Input**: One-hot encoded amino acid sequences
- **Output**: 3D coordinates for backbone atoms (N, CA, C, O) for each residue

### Visualization

The visualization includes:
- Alpha helices shown in red
- Beta sheets shown in yellow  
- Loops/coils shown in green
- Multiple viewing angles for better structure analysis

## Troubleshooting

### Common Issues

1. **ImportError with Biopython**: If you encounter issues with `from Bio.PDB.Polypeptide import three_to_one`, use this alternative:
   ```python
   from Bio.Data.IUPACData import protein_letters_3to1_extended
   
   def three_to_one(residue_name):
       return protein_letters_3to1_extended.get(residue_name, "X")
   ```

2. **Matplotlib GUI Backend Errors**: If you see errors related to matplotlib backend:
   ```python
   # Add this at the start of your script
   import matplotlib
   matplotlib.use('Agg')  # Use non-interactive backend
   ```

3. **Visualization Size Mismatch**: If you encounter a size mismatch between sequence length and coordinate size, ensure your visualization code handles this properly.


## License

This project is licensed under the MIT License - see the LICENSE file for details.