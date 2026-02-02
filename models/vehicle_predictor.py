import torch
import torch.nn as nn

from .temporal_encoder import SpatioTemporalEncoder


class DensityEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, spatial_size=18):
        super().__init__()
        self.spatial_size = spatial_size
        self.embed_dim = embed_dim
        
        self.conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        
        self.pos_embed = nn.Parameter(torch.randn(1, spatial_size * spatial_size, embed_dim))
        
    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            features = self.conv_embed(x)
            features = features.flatten(2).transpose(1, 2)
            features = features + self.pos_embed
            features = features.view(B, T, H * W, -1)
        else:
            B, C, H, W = x.shape
            features = self.conv_embed(x)
            features = features.flatten(2).transpose(1, 2)
            features = features + self.pos_embed
        return features


class DensityDecoder(nn.Module):
    def __init__(self, embed_dim=64, spatial_size=18, out_channels=1):
        super().__init__()
        self.spatial_size = spatial_size
        
        self.decode = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_channels)
        )
        
    def forward(self, x):
        B, N, D = x.shape
        out = self.decode(x)
        out = out.transpose(1, 2).view(B, -1, self.spatial_size, self.spatial_size)
        return out


class MultiHorizonDensityHead(nn.Module):
    def __init__(self, embed_dim, spatial_size, prediction_horizons):
        super().__init__()
        self.prediction_horizons = prediction_horizons
        self.spatial_size = spatial_size
        
        self.horizon_decoders = nn.ModuleList([
            DensityDecoder(embed_dim, spatial_size, out_channels=1)
            for _ in prediction_horizons
        ])
        
    def forward(self, encoded_features):
        B, T, N, D = encoded_features.shape
        last_features = encoded_features[:, -1, :, :]
        
        predictions = []
        for decoder in self.horizon_decoders:
            pred = decoder(last_features)
            predictions.append(pred)
            
        return torch.stack(predictions, dim=1)


class VehiclePredictor(nn.Module):
    def __init__(
        self,
        in_channels=1,
        embed_dim=64,
        spatial_size=18,
        num_encoder_layers=4,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        prediction_horizons=None,
        max_seq_len=50
    ):
        super().__init__()
        
        if prediction_horizons is None:
            prediction_horizons = [1, 3, 5, 10, 15]
            
        self.spatial_size = spatial_size
        self.prediction_horizons = prediction_horizons
        
        self.encoder = DensityEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            spatial_size=spatial_size
        )
        
        self.temporal_encoder = SpatioTemporalEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        self.prediction_head = MultiHorizonDensityHead(
            embed_dim=embed_dim,
            spatial_size=spatial_size,
            prediction_horizons=prediction_horizons
        )
        
    def forward(self, x):
        features = self.encoder(x)
        encoded = self.temporal_encoder(features)
        predictions = self.prediction_head(encoded)
        return predictions


class VehiclePredictorLite(nn.Module):
    def __init__(
        self,
        in_channels=1,
        embed_dim=32,
        spatial_size=18,
        hidden_dim=64,
        num_layers=2,
        prediction_horizons=None
    ):
        super().__init__()
        
        if prediction_horizons is None:
            prediction_horizons = [1, 3, 5]
            
        self.spatial_size = spatial_size
        self.prediction_horizons = prediction_horizons
        num_patches = spatial_size * spatial_size
        
        self.encoder = DensityEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            spatial_size=spatial_size
        )
        
        self.temporal_encoder = nn.LSTM(
            input_size=embed_dim * num_patches,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, num_patches),
                nn.Unflatten(-1, (1, spatial_size, spatial_size))
            )
            for _ in prediction_horizons
        ])
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        features = self.encoder(x)
        features_flat = features.view(B, T, -1)
        
        lstm_out, _ = self.temporal_encoder(features_flat)
        last_hidden = lstm_out[:, -1, :]
        
        predictions = []
        for head in self.prediction_heads:
            pred = head(last_hidden)
            predictions.append(pred)
            
        return torch.stack(predictions, dim=1)
