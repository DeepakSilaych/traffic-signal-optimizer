import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, T, N, D = x.shape
        x_flat = x.view(B * T, N, D)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = self.norm(x_flat + attn_out)
        return attn_out.view(B, T, N, D)


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        B, T, N, D = x.shape
        x_transposed = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        attn_out, _ = self.attn(x_transposed, x_transposed, x_transposed, attn_mask=mask)
        attn_out = self.norm(x_transposed + attn_out)
        return attn_out.view(B, N, T, D).permute(0, 2, 1, 3)


class SpatioTemporalBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.spatial_attn = SpatialAttention(embed_dim, num_heads, dropout)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, dropout)
        
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, temporal_mask=None):
        x = self.spatial_attn(x)
        x = self.temporal_attn(x, temporal_mask)
        x = self.norm(x + self.mlp(x))
        return x


class SpatioTemporalEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        mlp_ratio=4,
        dropout=0.1,
        max_seq_len=100
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        self.layers = nn.ModuleList([
            SpatioTemporalBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, temporal_mask=None):
        B, T, N, D = x.shape
        
        x_pos = x.view(B * N, T, D)
        x_pos = self.pos_encoder(x_pos)
        x = x_pos.view(B, T, N, D)
        
        for layer in self.layers:
            x = layer(x, temporal_mask)
            
        return self.norm(x)


class TemporalOnlyEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        mlp_ratio=4,
        dropout=0.1,
        max_seq_len=100
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        B, T, N, D = x.shape
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        x_flat = self.pos_encoder(x_flat)
        encoded = self.encoder(x_flat, mask=mask)
        encoded = self.norm(encoded)
        return encoded.view(B, N, T, D).permute(0, 2, 1, 3)
