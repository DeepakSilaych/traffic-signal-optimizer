import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class VehiclePredictionDataset(Dataset):
    def __init__(
        self,
        density_maps,
        vehicle_counts,
        seq_len=24,
        prediction_horizons=None,
        transform=None
    ):
        self.density_maps = density_maps
        self.vehicle_counts = vehicle_counts
        self.seq_len = seq_len
        self.prediction_horizons = prediction_horizons or [1, 3, 5, 10, 15]
        self.transform = transform
        
        self.max_horizon = max(self.prediction_horizons)
        self.valid_indices = len(density_maps) - seq_len - self.max_horizon
        
    def __len__(self):
        return max(0, self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.seq_len
        
        input_maps = self.density_maps[start_idx:end_idx]
        
        target_counts = []
        for horizon in self.prediction_horizons:
            target_idx = end_idx + horizon - 1
            target_counts.append(self.vehicle_counts[target_idx])
        target_counts = np.stack(target_counts, axis=0)
        
        if self.transform:
            input_maps = self.transform(input_maps)
            
        input_maps = torch.FloatTensor(input_maps)
        target_counts = torch.FloatTensor(target_counts)
        
        if input_maps.dim() == 3:
            input_maps = input_maps.unsqueeze(1)
            
        return input_maps, target_counts


class PrecomputedDensityDataset(Dataset):
    def __init__(
        self,
        density_dir,
        seq_len=24,
        prediction_horizons=None,
        num_zones=8
    ):
        self.density_dir = Path(density_dir)
        self.seq_len = seq_len
        self.prediction_horizons = prediction_horizons or [1, 3, 5, 10, 15]
        self.num_zones = num_zones
        self.max_horizon = max(self.prediction_horizons)
        
        self.density_files = sorted(self.density_dir.glob('*.npy'))
        self.valid_indices = len(self.density_files) - seq_len - self.max_horizon
        
        self._compute_zone_counts()
        
    def _compute_zone_counts(self):
        self.zone_counts = []
        for f in self.density_files:
            density = np.load(str(f))
            counts = self._extract_zone_counts(density)
            self.zone_counts.append(counts)
        self.zone_counts = np.array(self.zone_counts)
        
    def _extract_zone_counts(self, density_map):
        h, w = density_map.shape[-2:]
        zones_per_row = int(np.sqrt(self.num_zones))
        zone_h = h // zones_per_row
        zone_w = w // zones_per_row
        
        counts = []
        for i in range(zones_per_row):
            for j in range(zones_per_row):
                y1, y2 = i * zone_h, (i + 1) * zone_h
                x1, x2 = j * zone_w, (j + 1) * zone_w
                counts.append(density_map[..., y1:y2, x1:x2].sum())
                
        return np.array(counts[:self.num_zones])
        
    def __len__(self):
        return max(0, self.valid_indices)
    
    def __getitem__(self, idx):
        input_maps = []
        for i in range(idx, idx + self.seq_len):
            density = np.load(str(self.density_files[i]))
            input_maps.append(density)
        input_maps = np.stack(input_maps, axis=0)
        
        target_counts = []
        for horizon in self.prediction_horizons:
            target_idx = idx + self.seq_len + horizon - 1
            target_counts.append(self.zone_counts[target_idx])
        target_counts = np.stack(target_counts, axis=0)
        
        input_maps = torch.FloatTensor(input_maps)
        target_counts = torch.FloatTensor(target_counts)
        
        if input_maps.dim() == 3:
            input_maps = input_maps.unsqueeze(1)
            
        return input_maps, target_counts


class MultiIntersectionDataset(Dataset):
    def __init__(
        self,
        intersection_data,
        seq_len=24,
        prediction_horizons=None,
        transform=None
    ):
        self.intersection_data = intersection_data
        self.seq_len = seq_len
        self.prediction_horizons = prediction_horizons or [1, 3, 5, 10, 15]
        self.transform = transform
        self.max_horizon = max(self.prediction_horizons)
        
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self):
        samples = []
        for intersection_id, data in self.intersection_data.items():
            density_maps = data['density_maps']
            vehicle_counts = data['vehicle_counts']
            
            valid_len = len(density_maps) - self.seq_len - self.max_horizon
            for idx in range(valid_len):
                samples.append((intersection_id, idx))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        intersection_id, start_idx = self.samples[idx]
        data = self.intersection_data[intersection_id]
        
        end_idx = start_idx + self.seq_len
        input_maps = data['density_maps'][start_idx:end_idx]
        
        target_counts = []
        for horizon in self.prediction_horizons:
            target_idx = end_idx + horizon - 1
            target_counts.append(data['vehicle_counts'][target_idx])
        target_counts = np.stack(target_counts, axis=0)
        
        if self.transform:
            input_maps = self.transform(input_maps)
            
        input_maps = torch.FloatTensor(input_maps)
        target_counts = torch.FloatTensor(target_counts)
        
        if input_maps.dim() == 3:
            input_maps = input_maps.unsqueeze(1)
            
        return input_maps, target_counts, intersection_id


def create_synthetic_dataset(
    num_samples=1000,
    spatial_size=18,
    num_zones=8,
    seq_len=24,
    prediction_horizons=None
):
    prediction_horizons = prediction_horizons or [1, 3, 5, 10, 15]
    max_horizon = max(prediction_horizons)
    total_len = num_samples + seq_len + max_horizon
    
    t = np.linspace(0, 100, total_len)
    base_signal = np.sin(t * 0.1) * 0.3 + 0.5
    base_signal += np.sin(t * 0.5) * 0.1
    noise = np.random.randn(total_len) * 0.05
    base_signal = np.clip(base_signal + noise, 0, 1)
    
    density_maps = np.zeros((total_len, spatial_size, spatial_size))
    for i in range(total_len):
        base_map = np.random.rand(spatial_size, spatial_size) * 0.1
        for zone in range(num_zones):
            cx = (zone % 4) * (spatial_size // 4) + spatial_size // 8
            cy = (zone // 4) * (spatial_size // 2) + spatial_size // 4
            
            y, x = np.ogrid[:spatial_size, :spatial_size]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            zone_density = np.exp(-dist**2 / (2 * 3**2)) * base_signal[i]
            density_maps[i] += zone_density
            
    density_maps = np.clip(density_maps, 0, 1)
    
    vehicle_counts = np.zeros((total_len, num_zones))
    for i in range(total_len):
        for zone in range(num_zones):
            cx = (zone % 4) * (spatial_size // 4) + spatial_size // 8
            cy = (zone // 4) * (spatial_size // 2) + spatial_size // 4
            
            x_start = max(0, cx - 3)
            x_end = min(spatial_size, cx + 4)
            y_start = max(0, cy - 3)
            y_end = min(spatial_size, cy + 4)
            
            vehicle_counts[i, zone] = density_maps[i, y_start:y_end, x_start:x_end].sum()
    
    return VehiclePredictionDataset(
        density_maps=density_maps,
        vehicle_counts=vehicle_counts,
        seq_len=seq_len,
        prediction_horizons=prediction_horizons
    )
