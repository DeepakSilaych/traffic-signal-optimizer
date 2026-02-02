import torch
import torch.nn as nn
from collections import deque
from typing import Optional, Dict
import numpy as np


def load_estimation_model(checkpoint_path: Optional[str] = None) -> nn.Module:
    from models import REsnext
    model = REsnext()
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def load_prediction_model(checkpoint_path: Optional[str] = None, config: Optional[Dict] = None) -> nn.Module:
    from models import VehiclePredictor
    config = config or {}
    model = VehiclePredictor(
        embed_dim=config.get('embed_dim', 64),
        spatial_size=config.get('spatial_size', 18),
        prediction_horizons=config.get('prediction_horizons', [1, 3, 5, 10, 15])
    )
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


class TrafficPredictor:
    def __init__(
        self,
        estimation_checkpoint: Optional[str] = None,
        prediction_checkpoint: Optional[str] = None,
        buffer_size: int = 24,
        device: str = 'cpu'
    ):
        self.device = device
        self.buffer_size = buffer_size
        
        self.estimation_model = load_estimation_model(estimation_checkpoint).to(device)
        self.prediction_model = load_prediction_model(prediction_checkpoint).to(device)
        
        self.density_buffer = deque(maxlen=buffer_size)
        
    @torch.no_grad()
    def process_frame(self, frame: np.ndarray) -> Dict:
        if frame.shape != (72, 72, 3):
            import cv2
            frame = cv2.resize(frame, (72, 72))
            
        x = torch.FloatTensor(frame).permute(2, 0, 1).unsqueeze(0) / 255.0
        x = x.to(self.device)
        
        density = self.estimation_model(x)
        self.density_buffer.append(density.cpu())
        
        current_count = density.sum().item()
        
        result = {
            'density_map': density.cpu().numpy()[0],
            'current_count': current_count,
            'buffer_ready': len(self.density_buffer) >= self.buffer_size
        }
        
        if len(self.density_buffer) >= self.buffer_size:
            seq = torch.stack(list(self.density_buffer), dim=1).to(self.device)
            future_densities = self.prediction_model(seq)
            
            result['future_densities'] = future_densities.cpu().numpy()[0]
            result['future_counts'] = future_densities.sum(dim=(2, 3, 4)).cpu().numpy()[0]
            
        return result
    
    def reset(self):
        self.density_buffer.clear()


if __name__ == '__main__':
    predictor = TrafficPredictor()
    
    for i in range(30):
        dummy_frame = np.random.randint(0, 255, (72, 72, 3), dtype=np.uint8)
        result = predictor.process_frame(dummy_frame)
        
        print(f"Frame {i+1}: Count={result['current_count']:.1f}", end='')
        if result['buffer_ready']:
            print(f" | Future counts: {result['future_counts']}")
        else:
            print(f" | Buffering... ({len(predictor.density_buffer)}/{predictor.buffer_size})")
