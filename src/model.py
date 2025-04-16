import torch
import torch.nn as nn
import math

# Constants
MAX_SEQUENCE_LENGTH = 128  # Maximum protein sequence length
NUM_AMINO_ACIDS = 20      # Number of standard amino acids

# Sinusoidal embedding for diffusion timestep
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Self-attention mechanism for protein features
class ProteinSelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.heads, d // self.heads).transpose(1, 2), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        out = self.to_out(out)
        
        return out

# Sequence encoder for amino acid sequences
class SequenceEncoder(nn.Module):
    def __init__(self, num_amino_acids, dim):
        super().__init__()
        self.embedding = nn.Linear(num_amino_acids, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, MAX_SEQUENCE_LENGTH, dim))
        
    def forward(self, seq):
        # seq shape: [batch_size, sequence_length, num_amino_acids]
        x = self.embedding(seq)
        x = x + self.position_embedding
        return x

# Protein structure denoiser
class ProteinDenoiser(nn.Module):
    def __init__(self, hidden_dims, diffusion_time_embedding_dim=256, n_times=1000):
        super(ProteinDenoiser, self).__init__()
        
        # Time embedding
        self.time_embedding = SinusoidalPosEmb(diffusion_time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_time_embedding_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        
        # Sequence encoder
        self.sequence_encoder = SequenceEncoder(NUM_AMINO_ACIDS, hidden_dims[0])
        
        # Initial projection for coordinates
        self.coord_embed = nn.Sequential(
            nn.Linear(3, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            ProteinSelfAttention(hidden_dims[0])
            for _ in range(len(hidden_dims) // 2)
        ])
        
        # Residual blocks
        self.residual_layers = nn.ModuleList([])
        for i in range(len(hidden_dims)):
            self.residual_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[0]),
                nn.SiLU(),
                nn.Linear(hidden_dims[0], hidden_dims[0])
            ))
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dims[0], 3)
        
    def forward(self, coords, seq, diffusion_timestep):
        # coords shape: [batch_size, sequence_length, 4, 3] (4 backbone atoms per residue)
        # seq shape: [batch_size, sequence_length, num_amino_acids] (one-hot encoded)
        batch_size, seq_len, atoms_per_res, xyz_dim = coords.shape
        
        # Embed diffusion time
        diffusion_emb = self.time_embedding(diffusion_timestep)
        diffusion_emb = self.time_mlp(diffusion_emb)  # [batch_size, hidden_dim]
        diffusion_emb = diffusion_emb.unsqueeze(1).unsqueeze(1).repeat(1, seq_len, atoms_per_res, 1)
        
        # Embed sequence
        seq_emb = self.sequence_encoder(seq)  # [batch_size, sequence_length, hidden_dim]
        seq_emb = seq_emb.unsqueeze(2).repeat(1, 1, atoms_per_res, 1)
        
        # Embed coordinates
        coords_flat = coords.reshape(batch_size, seq_len * atoms_per_res, xyz_dim)
        coords_emb = self.coord_embed(coords_flat)
        coords_emb = coords_emb.reshape(batch_size, seq_len, atoms_per_res, -1)
        
        # Combine embeddings
        x = coords_emb + seq_emb + diffusion_emb
        
        # Process through residual blocks and attention layers
        for i, (res_layer, attn_layer) in enumerate(zip(self.residual_layers, self.attention_layers + [None] * (len(self.residual_layers) - len(self.attention_layers)))):
            # Apply residual block
            res_out = res_layer(x)
            x = x + res_out
            
            # Apply attention if available
            if i < len(self.attention_layers):
                # Reshape for attention
                x_flat = x.reshape(batch_size, seq_len * atoms_per_res, -1)
                attn_out = attn_layer(x_flat)
                attn_out = attn_out.reshape(batch_size, seq_len, atoms_per_res, -1)
                x = x + attn_out
        
        # Project to 3D coordinates
        pred_noise = self.out_proj(x)
        
        return pred_noise