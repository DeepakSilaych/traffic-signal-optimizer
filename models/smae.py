import torch
import torch.nn as nn


class SpatialPatchEmbedding(nn.Module):
    def __init__(self, num_nodes, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)
        
    def forward(self, x):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = self.proj(x)
        return x


class STPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_t=100, max_n=400):
        super().__init__()
        self.temporal_embed = nn.Parameter(torch.randn(1, 1, max_t, embed_dim))
        self.spatial_embed = nn.Parameter(torch.randn(1, max_n, 1, embed_dim))
        
    def forward(self, x):
        B, N, T, D = x.shape
        x = x + self.temporal_embed[:, :, :T, :]
        x = x + self.spatial_embed[:, :N, :, :]
        return x


class SpatialTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        B, N, T, D = x.shape
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * T, N, D)
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.view(B, T, N, D).permute(0, 2, 1, 3).contiguous()


class SpatialEncoder(nn.Module):
    def __init__(self, num_nodes, in_channels, embed_dim, num_heads, depth, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.patch_embed = SpatialPatchEmbedding(num_nodes, in_channels, embed_dim)
        self.pos_embed = STPositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList([
            SpatialTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        
        if mask is not None:
            B, N, T, D = x.shape
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, -1, D)
            x = x * mask_expanded
            
        for block in self.blocks:
            x = block(x)
        return self.norm(x.mean(dim=2))


class SpatialDecoder(nn.Module):
    def __init__(self, num_nodes, out_channels, embed_dim, num_heads, depth, seq_len, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.out_channels = out_channels
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim))
        self.pos_embed = STPositionalEncoding(embed_dim)
        
        self.blocks = nn.ModuleList([
            SpatialTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_channels)
        
    def forward(self, encoded, mask, seq_len):
        B, N, D = encoded.shape
        
        x = encoded.unsqueeze(2).expand(-1, -1, seq_len, -1)
        mask_tokens = self.mask_token.expand(B, N, seq_len, -1)
        
        mask_t = mask.permute(0, 2, 1).contiguous()
        mask_expanded = mask_t.unsqueeze(-1).expand(-1, -1, -1, D)
        x = x * mask_expanded + mask_tokens * (1 - mask_expanded)
        
        x = self.pos_embed(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        x = self.proj(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


class SMAE(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_channels=1,
        embed_dim=96,
        encoder_depth=4,
        decoder_depth=1,
        num_heads=4,
        mlp_ratio=4,
        mask_ratio=0.75,
        dropout=0.1
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        
        self.encoder = SpatialEncoder(
            num_nodes, in_channels, embed_dim, num_heads, encoder_depth, mlp_ratio, dropout
        )
        self.decoder = SpatialDecoder(
            num_nodes, in_channels, embed_dim, num_heads, decoder_depth,
            seq_len=1, mlp_ratio=mlp_ratio, dropout=dropout
        )
        
    def random_masking(self, x):
        B, N, T, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        mask = torch.zeros(B, N, device=x.device)
        mask.scatter_(1, ids_shuffle[:, :len_keep], 1)
        mask = mask.unsqueeze(2).expand(-1, -1, T)
        
        return mask
        
    def forward(self, x):
        B, T, N, C = x.shape
        
        x_transposed = x.permute(0, 2, 1, 3).contiguous()
        
        mask = self.random_masking(x_transposed)
        
        encoded = self.encoder(x, mask)
        decoded = self.decoder(encoded, mask, T)
        
        loss = ((decoded - x) ** 2).mean()
        
        return loss, encoded
    
    def get_representation(self, x):
        with torch.no_grad():
            B, T, N, C = x.shape
            mask = torch.ones(B, N, T, device=x.device)
            encoded = self.encoder(x, mask)
        return encoded
