import torch
import torch.nn as nn

from .tmae import TMAE
from .smae import SMAE


class DownstreamSTPredictor(nn.Module):
    def __init__(self, num_nodes, in_channels, embed_dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_channels, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, T, N, C = x.shape
        x = self.embed(x)
        x = x.view(B, T * N, -1)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.view(B, T, N, -1)
        return x.mean(dim=(1, 2))


class DownstreamPredictor(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_channels,
        embed_dim,
        hidden_dim=256,
        prediction_horizons=None,
        dropout=0.1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.prediction_horizons = prediction_horizons or [1, 3, 5, 10, 15]
        
        self.temporal_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.spatial_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.st_predictor = DownstreamSTPredictor(
            num_nodes=num_nodes,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=2,
            dropout=dropout
        )
        
        self.st_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_nodes)
            )
            for _ in self.prediction_horizons
        ])
        
    def forward(self, x, temporal_repr, spatial_repr):
        t_feat = self.temporal_mlp(temporal_repr)
        s_feat = self.spatial_mlp(spatial_repr)
        
        st_repr = self.st_predictor(x)
        st_feat = self.st_mlp(st_repr)
        
        fused = torch.cat([t_feat, s_feat, st_feat], dim=-1)
        fused = self.fusion(fused)
        
        predictions = []
        for fc in self.fc_layers:
            pred = fc(fused)
            predictions.append(pred)
            
        return torch.stack(predictions, dim=1)


class STDMAE(nn.Module):
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
        hidden_dim=256,
        prediction_horizons=None,
        dropout=0.1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.prediction_horizons = prediction_horizons or [1, 3, 5, 10, 15]
        
        self.tmae = TMAE(
            num_nodes=num_nodes,
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            mask_ratio=mask_ratio,
            dropout=dropout
        )
        
        self.smae = SMAE(
            num_nodes=num_nodes,
            in_channels=in_channels,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            mask_ratio=mask_ratio,
            dropout=dropout
        )
        
        self.downstream = DownstreamPredictor(
            num_nodes=num_nodes,
            in_channels=in_channels,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            prediction_horizons=prediction_horizons,
            dropout=dropout
        )
        
    def pretrain_tmae(self, x):
        return self.tmae(x)
    
    def pretrain_smae(self, x):
        return self.smae(x)
    
    def forward(self, x):
        temporal_repr = self.tmae.get_representation(x)
        spatial_repr = self.smae.get_representation(x)
        
        temporal_repr = temporal_repr.mean(dim=1)
        spatial_repr = spatial_repr.mean(dim=1)
        
        predictions = self.downstream(x, temporal_repr, spatial_repr)
        
        return predictions
    
    def load_pretrained(self, tmae_path=None, smae_path=None):
        if tmae_path:
            ckpt = torch.load(tmae_path, map_location='cpu')
            self.tmae.load_state_dict(ckpt['model_state_dict'])
            
        if smae_path:
            ckpt = torch.load(smae_path, map_location='cpu')
            self.smae.load_state_dict(ckpt['model_state_dict'])
            
    def freeze_encoders(self):
        for param in self.tmae.parameters():
            param.requires_grad = False
        for param in self.smae.parameters():
            param.requires_grad = False


class STDMAEForDensityMap(nn.Module):
    def __init__(
        self,
        spatial_size=18,
        in_channels=1,
        patch_size=12,
        embed_dim=96,
        encoder_depth=4,
        decoder_depth=1,
        num_heads=4,
        mlp_ratio=4,
        mask_ratio=0.75,
        hidden_dim=256,
        prediction_horizons=None,
        dropout=0.1
    ):
        super().__init__()
        num_nodes = spatial_size * spatial_size
        self.spatial_size = spatial_size
        self.prediction_horizons = prediction_horizons or [1, 3, 5, 10, 15]
        
        self.stdmae = STDMAE(
            num_nodes=num_nodes,
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            mask_ratio=mask_ratio,
            hidden_dim=hidden_dim,
            prediction_horizons=prediction_horizons,
            dropout=dropout
        )
        
        self.density_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_nodes, num_nodes),
                nn.ReLU(),
                nn.Linear(num_nodes, num_nodes)
            )
            for _ in self.prediction_horizons
        ])
        
    def _reshape_input(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T, C, -1)
        x = x.permute(0, 1, 3, 2).contiguous()
        return x
    
    def _reshape_output(self, x):
        B, num_horizons, N = x.shape
        x = x.view(B, num_horizons, 1, self.spatial_size, self.spatial_size)
        return x
        
    def forward(self, x):
        x_flat = self._reshape_input(x)
        
        predictions = self.stdmae(x_flat)
        
        density_preds = []
        for i, decoder in enumerate(self.density_decoders):
            pred = decoder(predictions[:, i])
            density_preds.append(pred)
        density_preds = torch.stack(density_preds, dim=1)
        
        output = self._reshape_output(density_preds)
        return output
    
    def pretrain_tmae(self, x):
        x_flat = self._reshape_input(x)
        return self.stdmae.pretrain_tmae(x_flat)
    
    def pretrain_smae(self, x):
        x_flat = self._reshape_input(x)
        return self.stdmae.pretrain_smae(x_flat)
    
    def load_pretrained(self, tmae_path=None, smae_path=None):
        self.stdmae.load_pretrained(tmae_path, smae_path)
        
    def freeze_encoders(self):
        self.stdmae.freeze_encoders()
