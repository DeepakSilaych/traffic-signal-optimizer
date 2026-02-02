import torch
import torch.nn as nn
import torch.nn.functional as F


class ZoneAttentionPooling(nn.Module):
    def __init__(self, in_channels=1, num_zones=8, embed_dim=64, spatial_size=18):
        super().__init__()
        self.num_zones = num_zones
        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        
        self.zone_queries = nn.Parameter(torch.randn(1, num_zones, embed_dim))
        
        self.spatial_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        
        self.pos_embed = nn.Parameter(torch.randn(1, spatial_size * spatial_size, embed_dim))
        
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
        self.zone_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        spatial_features = self.spatial_embed(x)
        spatial_features = spatial_features.flatten(2).transpose(1, 2)
        spatial_features = spatial_features + self.pos_embed[:, :spatial_features.size(1), :]
        
        zone_queries = self.zone_queries.expand(B, -1, -1)
        
        zone_features, attn_weights = self.cross_attn(
            zone_queries, spatial_features, spatial_features
        )
        
        zone_features = self.zone_norm(zone_features)
        
        return zone_features, attn_weights


class ZoneFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, num_zones=8, embed_dim=64, spatial_size=18):
        super().__init__()
        self.zone_pooling = ZoneAttentionPooling(
            in_channels, num_zones, embed_dim, spatial_size
        )
        
    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            zone_features, attn_weights = self.zone_pooling(x_flat)
            zone_features = zone_features.view(B, T, -1, zone_features.size(-1))
            attn_weights = attn_weights.view(B, T, *attn_weights.shape[1:])
        else:
            zone_features, attn_weights = self.zone_pooling(x)
            
        return zone_features, attn_weights


class ZoneCountRegressor(nn.Module):
    def __init__(self, embed_dim=64, num_zones=8):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
    def forward(self, zone_features):
        return self.regressor(zone_features).squeeze(-1)
