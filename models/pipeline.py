import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any

from .vehicle_estimation import REsnext
from .vehicle_predictor import VehiclePredictor, VehiclePredictorLite


class TrafficSignalPipeline(nn.Module):
    def __init__(
        self,
        estimation_model: Optional[nn.Module] = None,
        prediction_model: Optional[nn.Module] = None,
        freeze_estimation: bool = True
    ):
        super().__init__()
        
        self.estimation_model = estimation_model or REsnext()
        self.prediction_model = prediction_model
        
        if freeze_estimation:
            for param in self.estimation_model.parameters():
                param.requires_grad = False
                
    def estimate_density(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            images_flat = images.view(B * T, C, H, W)
            density_maps = self.estimation_model(images_flat)
            density_maps = density_maps.view(B, T, *density_maps.shape[1:])
        else:
            density_maps = self.estimation_model(images)
        return density_maps
    
    def predict_future(
        self,
        density_maps: torch.Tensor,
        mode: str = 'multi_horizon',
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        if self.prediction_model is None:
            raise ValueError("Prediction model not initialized")
        predictions, attn = self.prediction_model(density_maps, mode=mode, num_steps=num_steps)
        return predictions, attn
    
    def forward(
        self,
        images: torch.Tensor,
        mode: str = 'multi_horizon',
        num_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        density_maps = self.estimate_density(images)
        predictions, attn = self.predict_future(density_maps, mode, num_steps)
        
        return {
            'density_maps': density_maps,
            'predictions': predictions,
            'zone_attention': attn
        }


class InferencePipeline:
    def __init__(
        self,
        estimation_checkpoint: Optional[str] = None,
        prediction_checkpoint: Optional[str] = None,
        prediction_config: Optional[Dict[str, Any]] = None,
        device: str = 'cuda'
    ):
        self.device = device
        
        self.estimation_model = REsnext()
        if estimation_checkpoint:
            self._load_checkpoint(self.estimation_model, estimation_checkpoint)
        self.estimation_model.to(device).eval()
        
        if prediction_config:
            lite = prediction_config.get('lite', False)
            if lite:
                self.prediction_model = VehiclePredictorLite(
                    in_channels=prediction_config.get('in_channels', 1),
                    num_zones=prediction_config.get('num_zones', 8),
                    embed_dim=prediction_config.get('embed_dim', 32),
                    spatial_size=prediction_config.get('spatial_size', 18),
                    hidden_dim=prediction_config.get('hidden_dim', 64),
                    num_layers=prediction_config.get('num_layers', 2),
                    prediction_horizons=prediction_config.get('prediction_horizons', [1, 3, 5])
                )
            else:
                self.prediction_model = VehiclePredictor(
                    in_channels=prediction_config.get('in_channels', 1),
                    num_zones=prediction_config.get('num_zones', 8),
                    embed_dim=prediction_config.get('embed_dim', 64),
                    spatial_size=prediction_config.get('spatial_size', 18),
                    num_encoder_layers=prediction_config.get('num_encoder_layers', 4),
                    num_heads=prediction_config.get('num_heads', 4),
                    mlp_ratio=prediction_config.get('mlp_ratio', 4),
                    dropout=prediction_config.get('dropout', 0.1),
                    prediction_horizons=prediction_config.get('prediction_horizons', [1, 3, 5, 10, 15]),
                    max_seq_len=prediction_config.get('max_seq_len', 50)
                )
            if prediction_checkpoint:
                self._load_checkpoint(self.prediction_model, prediction_checkpoint)
            self.prediction_model.to(device).eval()
        else:
            self.prediction_model = None
            
        self.density_buffer: List[torch.Tensor] = []
        self.buffer_size = 24
        
    def _load_checkpoint(self, model: nn.Module, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
    @torch.no_grad()
    def process_frame(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        
        density_map = self.estimation_model(image)
        
        self.density_buffer.append(density_map.cpu())
        if len(self.density_buffer) > self.buffer_size:
            self.density_buffer.pop(0)
            
        result = {
            'density_map': density_map,
            'vehicle_count': density_map.sum().item()
        }
        
        if self.prediction_model and len(self.density_buffer) >= self.buffer_size:
            history = torch.stack(self.density_buffer, dim=1).to(self.device)
            predictions, attn = self.prediction_model(history, mode='multi_horizon')
            result['predictions'] = predictions
            result['zone_attention'] = attn
            
        return result
    
    def reset_buffer(self):
        self.density_buffer.clear()
