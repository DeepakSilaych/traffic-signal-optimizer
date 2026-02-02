import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class VehiclePredictionDataset(Dataset):
    def __init__(
        self,
        density_maps,
        seq_len=24,
        prediction_horizons=None
    ):
        self.density_maps = density_maps
        self.seq_len = seq_len
        self.prediction_horizons = prediction_horizons or [1, 3, 5, 10, 15]
        
        self.max_horizon = max(self.prediction_horizons)
        self.valid_indices = len(density_maps) - seq_len - self.max_horizon
        
    def __len__(self):
        return max(0, self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.seq_len
        
        input_maps = self.density_maps[start_idx:end_idx]
        
        target_maps = []
        for horizon in self.prediction_horizons:
            target_idx = end_idx + horizon - 1
            target_maps.append(self.density_maps[target_idx])
        target_maps = np.stack(target_maps, axis=0)
        
        input_maps = torch.FloatTensor(input_maps)
        target_maps = torch.FloatTensor(target_maps)
        
        if input_maps.dim() == 3:
            input_maps = input_maps.unsqueeze(1)
        if target_maps.dim() == 3:
            target_maps = target_maps.unsqueeze(2)
            
        return input_maps, target_maps


class PrecomputedDensityDataset(Dataset):
    def __init__(
        self,
        density_dir,
        seq_len=24,
        prediction_horizons=None
    ):
        self.density_dir = Path(density_dir)
        self.seq_len = seq_len
        self.prediction_horizons = prediction_horizons or [1, 3, 5, 10, 15]
        self.max_horizon = max(self.prediction_horizons)
        
        self.density_files = sorted(self.density_dir.glob('*.npy'))
        self.valid_indices = len(self.density_files) - seq_len - self.max_horizon
        
    def __len__(self):
        return max(0, self.valid_indices)
    
    def __getitem__(self, idx):
        input_maps = []
        for i in range(idx, idx + self.seq_len):
            density = np.load(str(self.density_files[i]))
            input_maps.append(density)
        input_maps = np.stack(input_maps, axis=0)
        
        target_maps = []
        for horizon in self.prediction_horizons:
            target_idx = idx + self.seq_len + horizon - 1
            target = np.load(str(self.density_files[target_idx]))
            target_maps.append(target)
        target_maps = np.stack(target_maps, axis=0)
        
        input_maps = torch.FloatTensor(input_maps)
        target_maps = torch.FloatTensor(target_maps)
        
        if input_maps.dim() == 3:
            input_maps = input_maps.unsqueeze(1)
        if target_maps.dim() == 3:
            target_maps = target_maps.unsqueeze(2)
            
        return input_maps, target_maps


def create_synthetic_dataset(
    num_samples=1000,
    spatial_size=18,
    seq_len=24,
    prediction_horizons=None
):
    from scipy.ndimage import gaussian_filter
    
    prediction_horizons = prediction_horizons or [1, 3, 5, 10, 15]
    max_horizon = max(prediction_horizons)
    total_len = num_samples + seq_len + max_horizon
    
    t = np.linspace(0, 100, total_len)
    base_signal = np.sin(t * 0.1) * 0.3 + 0.5
    base_signal += np.sin(t * 0.5) * 0.1
    noise = np.random.randn(total_len) * 0.05
    base_signal = np.clip(base_signal + noise, 0, 1)
    
    density_maps = np.zeros((total_len, spatial_size, spatial_size), dtype=np.float32)
    
    hotspots = [
        (spatial_size // 4, spatial_size // 4),
        (spatial_size // 4, 3 * spatial_size // 4),
        (3 * spatial_size // 4, spatial_size // 4),
        (3 * spatial_size // 4, 3 * spatial_size // 4),
    ]
    
    for i in range(total_len):
        for cy, cx in hotspots:
            y, x = np.ogrid[:spatial_size, :spatial_size]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            density_maps[i] += np.exp(-dist**2 / (2 * 3**2)) * base_signal[i]
            
        density_maps[i] += np.random.rand(spatial_size, spatial_size) * 0.05
        density_maps[i] = gaussian_filter(density_maps[i], sigma=1.0)
        
    density_maps = np.clip(density_maps, 0, 1)
    
    return VehiclePredictionDataset(
        density_maps=density_maps,
        seq_len=seq_len,
        prediction_horizons=prediction_horizons
    )
