import torch
import torch.nn as nn

from .zone_extractor import ZoneFeatureExtractor
from .temporal_encoder import SpatioTemporalEncoder


class MultiHorizonHead(nn.Module):
    def __init__(self, embed_dim, num_zones, prediction_horizons):
        super().__init__()
        self.prediction_horizons = prediction_horizons
        self.num_horizons = len(prediction_horizons)
        
        self.horizon_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 1)
            )
            for _ in prediction_horizons
        ])
        
    def forward(self, encoded_features):
        B, T, N, D = encoded_features.shape
        last_features = encoded_features[:, -1, :, :]
        
        predictions = []
        for proj in self.horizon_projections:
            pred = proj(last_features).squeeze(-1)
            predictions.append(pred)
            
        return torch.stack(predictions, dim=1)


class AutoregressiveHead(nn.Module):
    def __init__(self, embed_dim, num_zones, max_horizon=12):
        super().__init__()
        self.max_horizon = max_horizon
        self.embed_dim = embed_dim
        
        self.step_embedding = nn.Embedding(max_horizon, embed_dim)
        
        self.decoder = nn.GRU(
            input_size=embed_dim + num_zones,
            hidden_size=embed_dim,
            num_layers=2,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(embed_dim, num_zones)
        
    def forward(self, encoded_features, num_steps):
        B, T, N, D = encoded_features.shape
        
        context = encoded_features[:, -1, :, :].mean(dim=1)
        hidden = context.unsqueeze(0).repeat(2, 1, 1)
        
        last_pred = torch.zeros(B, N, device=encoded_features.device)
        predictions = []
        
        for step in range(num_steps):
            step_embed = self.step_embedding(
                torch.tensor([step], device=encoded_features.device)
            ).expand(B, -1)
            
            decoder_input = torch.cat([step_embed, last_pred], dim=-1).unsqueeze(1)
            
            output, hidden = self.decoder(decoder_input, hidden)
            pred = self.output_proj(output.squeeze(1))
            predictions.append(pred)
            last_pred = pred
            
        return torch.stack(predictions, dim=1)


class VehiclePredictor(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_zones=8,
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
            
        self.num_zones = num_zones
        self.prediction_horizons = prediction_horizons
        
        self.zone_extractor = ZoneFeatureExtractor(
            in_channels=in_channels,
            num_zones=num_zones,
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
        
        self.prediction_head = MultiHorizonHead(
            embed_dim=embed_dim,
            num_zones=num_zones,
            prediction_horizons=prediction_horizons
        )
        
        self.autoregressive_head = AutoregressiveHead(
            embed_dim=embed_dim,
            num_zones=num_zones,
            max_horizon=max(prediction_horizons) + 5
        )
        
    def forward(self, x, mode='multi_horizon', num_steps=None):
        zone_features, attn_weights = self.zone_extractor(x)
        encoded = self.temporal_encoder(zone_features)
        
        if mode == 'multi_horizon':
            predictions = self.prediction_head(encoded)
        elif mode == 'autoregressive':
            if num_steps is None:
                num_steps = max(self.prediction_horizons)
            predictions = self.autoregressive_head(encoded, num_steps)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        return predictions, attn_weights
    
    def get_zone_attention(self, x):
        _, attn_weights = self.zone_extractor(x)
        return attn_weights


class VehiclePredictorLite(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_zones=8,
        embed_dim=32,
        spatial_size=18,
        hidden_dim=64,
        num_layers=2,
        prediction_horizons=None
    ):
        super().__init__()
        
        if prediction_horizons is None:
            prediction_horizons = [1, 3, 5]
            
        self.num_zones = num_zones
        self.prediction_horizons = prediction_horizons
        
        self.zone_extractor = ZoneFeatureExtractor(
            in_channels=in_channels,
            num_zones=num_zones,
            embed_dim=embed_dim,
            spatial_size=spatial_size
        )
        
        self.temporal_encoder = nn.LSTM(
            input_size=embed_dim * num_zones,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.prediction_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_zones)
            for _ in prediction_horizons
        ])
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        zone_features, attn = self.zone_extractor(x)
        
        zone_flat = zone_features.view(B, T, -1)
        
        lstm_out, _ = self.temporal_encoder(zone_flat)
        last_hidden = lstm_out[:, -1, :]
        
        predictions = []
        for head in self.prediction_heads:
            pred = head(last_hidden)
            predictions.append(pred)
            
        return torch.stack(predictions, dim=1), attn
