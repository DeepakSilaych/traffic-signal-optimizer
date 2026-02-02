import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter


class VehicleEstimationDataset(Dataset):
    def __init__(
        self,
        images,
        density_maps,
        transform=None
    ):
        self.images = images
        self.density_maps = density_maps
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        density_map = self.density_maps[idx]
        
        if self.transform:
            image = self.transform(image)
            
        image = torch.FloatTensor(image)
        density_map = torch.FloatTensor(density_map)
        
        if image.dim() == 2:
            image = image.unsqueeze(0)
        elif image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
            
        if density_map.dim() == 2:
            density_map = density_map.unsqueeze(0)
            
        return image, density_map


class VehicleEstimationFromFiles(Dataset):
    def __init__(
        self,
        image_dir,
        density_dir,
        image_ext='.jpg',
        density_ext='.npy',
        transform=None
    ):
        self.image_dir = Path(image_dir)
        self.density_dir = Path(density_dir)
        self.transform = transform
        
        self.image_files = sorted(self.image_dir.glob(f'*{image_ext}'))
        self.density_files = sorted(self.density_dir.glob(f'*{density_ext}'))
        
        assert len(self.image_files) == len(self.density_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.density_files)} density maps"
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        import cv2
        
        image = cv2.imread(str(self.image_files[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (72, 72))
        image = image.astype(np.float32) / 255.0
        
        density_map = np.load(str(self.density_files[idx]))
        if density_map.shape != (18, 18):
            from scipy.ndimage import zoom
            scale = (18 / density_map.shape[0], 18 / density_map.shape[1])
            density_map = zoom(density_map, scale, order=1)
            
        if self.transform:
            image = self.transform(image)
            
        image = torch.FloatTensor(image).permute(2, 0, 1)
        density_map = torch.FloatTensor(density_map).unsqueeze(0)
        
        return image, density_map


def create_synthetic_estimation_data(
    num_samples=1000,
    image_size=72,
    density_size=18,
    max_vehicles=50
):
    images = []
    density_maps = []
    
    for _ in range(num_samples):
        image = np.random.rand(image_size, image_size, 3).astype(np.float32) * 0.3
        
        density_map = np.zeros((density_size, density_size), dtype=np.float32)
        
        num_vehicles = np.random.randint(5, max_vehicles)
        
        for _ in range(num_vehicles):
            x = np.random.randint(0, density_size)
            y = np.random.randint(0, density_size)
            
            density_map[y, x] += 1
            
            img_x = int(x * image_size / density_size)
            img_y = int(y * image_size / density_size)
            
            size = np.random.randint(2, 5)
            color = np.random.rand(3) * 0.5 + 0.5
            
            x1 = max(0, img_x - size)
            x2 = min(image_size, img_x + size)
            y1 = max(0, img_y - size)
            y2 = min(image_size, img_y + size)
            
            image[y1:y2, x1:x2] = color
            
        density_map = gaussian_filter(density_map, sigma=1.5)
        
        images.append(image)
        density_maps.append(density_map)
        
    return VehicleEstimationDataset(
        images=np.array(images),
        density_maps=np.array(density_maps)
    )
