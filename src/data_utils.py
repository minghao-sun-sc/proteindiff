import os
import torch
import numpy as np
from Bio.PDB import PDBParser, PDBIO
from tqdm import tqdm

# Constants for protein structure
MAX_SEQUENCE_LENGTH = 128  # Maximum protein sequence length
NUM_AMINO_ACIDS = 20      # Number of standard amino acids
MAX_ATOMS_PER_RESIDUE = 14  # Maximum number of atoms per amino acid residue

def one_hot_encode_sequence(sequence):
    """Convert amino acid sequence to one-hot encoding"""
    # Map each amino acid to a number
    aa_dict = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
               'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
    
    # Create one-hot encoding
    encoding = torch.zeros(len(sequence), NUM_AMINO_ACIDS)
    for i, aa in enumerate(sequence):
        if aa in aa_dict:
            encoding[i, aa_dict[aa]] = 1.0
    
    # Pad if necessary
    if len(sequence) < MAX_SEQUENCE_LENGTH:
        padding = torch.zeros(MAX_SEQUENCE_LENGTH - len(sequence), NUM_AMINO_ACIDS)
        encoding = torch.cat([encoding, padding], dim=0)
    else:
        encoding = encoding[:MAX_SEQUENCE_LENGTH]
        
    return encoding

def load_protein_structure(pdb_file):
    """Load protein structure from PDB file and extract coordinates"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    # Extract coordinates and sequence
    coords = []
    sequence = ""
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":  # Standard amino acid
                    resname = residue.get_resname()
                    # Convert 3-letter code to 1-letter code
                    # from Bio.PDB.Polypeptide import aa3to1 as three_to_one
                    # Replace the Bio.PDB.Polypeptide import with this code block
                    from Bio.Data.IUPACData import protein_letters_3to1_extended

                    def three_to_one(residue_name):
                        """Convert three letter amino acid code to one letter code"""
                        return protein_letters_3to1_extended.get(residue_name.upper(), 'X')
                    try:
                        aa = three_to_one(resname)
                        sequence += aa
                    except:
                        sequence += "X"  # Unknown amino acid
                    
                    # Get atom coordinates
                    residue_coords = []
                    for atom in residue:
                        if atom.get_name() in ["N", "CA", "C", "O"]:  # Focus on backbone atoms
                            residue_coords.append(atom.get_coord())
                    
                    # Pad if necessary
                    while len(residue_coords) < 4:  # Ensure we have coordinates for all backbone atoms
                        residue_coords.append(np.zeros(3))
                        
                    coords.append(residue_coords)
    
    # Convert to tensor and normalize
    coords = torch.tensor(coords, dtype=torch.float32)
    
    # Pad or truncate to fixed length
    if len(coords) < MAX_SEQUENCE_LENGTH:
        padding = torch.zeros(MAX_SEQUENCE_LENGTH - len(coords), 4, 3)
        coords = torch.cat([coords, padding], dim=0)
    else:
        coords = coords[:MAX_SEQUENCE_LENGTH]
    
    return coords, sequence

def save_protein_structure(coords, sequence, output_file):
    """Save generated coordinates as a PDB file"""
    # Convert to numpy
    coords = coords.detach().cpu().numpy()
    
    # Create a simple PDB file
    with open(output_file, 'w') as f:
        atom_index = 1
        for i, residue_coords in enumerate(coords):
            if i >= len(sequence):
                break
                
            for j, atom_name in enumerate(["N", "CA", "C", "O"]):
                if j < len(residue_coords):
                    x, y, z = residue_coords[j]
                    f.write(f"ATOM  {atom_index:5d}  {atom_name:<3s} {sequence[i]:3s} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]}\n")
                    atom_index += 1
        f.write("END\n")

# Dataset class for loading protein structures
class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pdb_files = [f for f in os.listdir(data_dir) if f.endswith('.pdb')]
        
    def __len__(self):
        return len(self.pdb_files)
    
    def __getitem__(self, idx):
        pdb_file = os.path.join(self.data_dir, self.pdb_files[idx])
        coords, sequence = load_protein_structure(pdb_file)
        sequence_encoding = one_hot_encode_sequence(sequence)
        
        return {
            'coords': coords,
            'sequence': sequence_encoding,
            'raw_sequence': sequence
        }

def download_example_pdb_files(output_dir="data/pdb_files"):
    """Download a few example PDB files for testing"""
    import requests
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # List of some small protein PDB IDs
    pdb_ids = ["1CRN", "1ENH", "1HZ6", "1LMB", "1ROP"]
    
    for pdb_id in tqdm(pdb_ids, desc="Downloading PDB files"):
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        output_file = os.path.join(output_dir, f"{pdb_id}.pdb")
        
        if not os.path.exists(output_file):
            response = requests.get(pdb_url)
            if response.status_code == 200:
                with open(output_file, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded {pdb_id}.pdb")
            else:
                print(f"Failed to download {pdb_id}.pdb")