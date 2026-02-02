import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * in_channels, embed_dim)
        
    def forward(self, x):
        B, T, N, C = x.shape
        num_patches = T // self.patch_size
        x = x.view(B, num_patches, self.patch_size, N, C)
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        x = x.view(B, N, num_patches, -1)
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


class TransformerBlock(nn.Module):
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
        x_flat = x.view(B * N, T, D)
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.view(B, N, T, D)


class TemporalEncoder(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim, num_heads, depth, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.pos_embed = STPositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
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


class TemporalDecoder(nn.Module):
    def __init__(self, patch_size, out_channels, embed_dim, num_heads, depth, num_patches, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_channels = out_channels
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim))
        self.pos_embed = STPositionalEncoding(embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, patch_size * out_channels)
        
    def forward(self, encoded, mask):
        B, N, D = encoded.shape
        
        x = encoded.unsqueeze(2).expand(-1, -1, self.num_patches, -1)
        mask_tokens = self.mask_token.expand(B, N, self.num_patches, -1)
        
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, -1, D)
        x = x * mask_expanded + mask_tokens * (1 - mask_expanded)
        
        x = self.pos_embed(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        x = self.proj(x)
        x = x.view(B, N, self.num_patches, self.patch_size, self.out_channels)
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        x = x.view(B, self.num_patches * self.patch_size, N, self.out_channels)
        return x


class TMAE(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_channels=1,
        patch_size=12,
        embed_dim=96,
        encoder_depth=4,
        decoder_depth=1,
        num_heads=4,
        mlp_ratio=4,
        mask_ratio=0.75,
        dropout=0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        
        self.encoder = TemporalEncoder(
            patch_size, in_channels, embed_dim, num_heads, encoder_depth, mlp_ratio, dropout
        )
        self.decoder = TemporalDecoder(
            patch_size, in_channels, embed_dim, num_heads, decoder_depth, 
            num_patches=1, mlp_ratio=mlp_ratio, dropout=dropout
        )
        
    def random_masking(self, x):
        B, N, T, D = x.shape
        num_patches = T
        len_keep = int(num_patches * (1 - self.mask_ratio))
        
        noise = torch.rand(B, N, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=2)
        
        mask = torch.zeros(B, N, num_patches, device=x.device)
        mask.scatter_(2, ids_shuffle[:, :, :len_keep], 1)
        
        return mask
        
    def forward(self, x):
        B, T, N, C = x.shape
        num_patches = T // self.patch_size
        self.decoder.num_patches = num_patches
        
        x_patches = x.view(B, num_patches, self.patch_size, N, C)
        x_patches = x_patches.permute(0, 3, 1, 2, 4).contiguous()
        x_patches = x_patches.view(B, N, num_patches, -1)
        
        mask = self.random_masking(x_patches)
        
        encoded = self.encoder(x, mask)
        decoded = self.decoder(encoded, mask)
        
        loss = ((decoded - x) ** 2).mean()
        
        return loss, encoded
    
    def get_representation(self, x):
        with torch.no_grad():
            B, T, N, C = x.shape
            num_patches = T // self.patch_size
            
            x_patches = x.view(B, num_patches, self.patch_size, N, C)
            x_patches = x_patches.permute(0, 3, 1, 2, 4).contiguous()
            x_patches = x_patches.view(B, N, num_patches, -1)
            
            mask = torch.ones(B, N, num_patches, device=x.device)
            encoded = self.encoder(x, mask)
            
        return encoded
