import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from architecture.vehicle_estimation_model import VehicleEstimationModel
from utils import DensityMapLoss


def load_data(image_dir, density_dir):
    image_files = sorted(Path(image_dir).glob('*'))
    density_files = sorted(Path(density_dir).glob('*.npy'))

    images = []
    for f in image_files:
        img = Image.open(f).convert('RGB').resize((72, 72))
        images.append(np.array(img, dtype=np.float32) / 255.0)

    densities = [np.load(str(f)) for f in density_files]

    images = torch.FloatTensor(np.stack(images)).permute(0, 3, 1, 2)
    densities = torch.FloatTensor(np.stack(densities)).unsqueeze(1)
    return TensorDataset(images, densities)


class EstimationTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=1e-3, weight_decay=1e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = DensityMapLoss(count_weight=0.1)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.best_val_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        total_loss, total_pixel, total_count, n = 0, 0, 0, 0
        for images, density_maps in self.train_loader:
            images, density_maps = images.to(self.device), density_maps.to(self.device)
            self.optimizer.zero_grad()
            loss, loss_dict = self.criterion(self.model(images), density_maps)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item()
            total_pixel += loss_dict['pixel_loss']
            total_count += loss_dict['count_loss']
            n += 1
        return {'total': total_loss / n, 'pixel': total_pixel / n, 'count': total_count / n}

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, total_mae, n = 0, 0, 0
        for images, density_maps in self.val_loader:
            images, density_maps = images.to(self.device), density_maps.to(self.device)
            predictions = self.model(images)
            loss, _ = self.criterion(predictions, density_maps)
            mae = torch.mean(torch.abs(
                predictions.sum(dim=(1, 2, 3)) - density_maps.sum(dim=(1, 2, 3))
            ))
            total_loss += loss.item()
            total_mae += mae.item()
            n += 1
        return {'loss': total_loss / n, 'mae': total_mae / n}

    def train(self, num_epochs, save_dir=None):
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(num_epochs):
            train_m = self.train_epoch()
            val_m = self.validate()
            self.scheduler.step()
            if val_m['loss'] < self.best_val_loss:
                self.best_val_loss = val_m['loss']
                if save_dir:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_val_loss': self.best_val_loss,
                    }, save_dir / 'best_model.pt')
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train={train_m['total']:.4f} | Val={val_m['loss']:.4f}, MAE={val_m['mae']:.2f}")


if __name__ == '__main__':
    image_dir = 'data/images'
    density_dir = 'data/density_maps'
    num_epochs = 3
    batch_size = 16
    lr = 1e-3
    save_dir = 'checkpoints/estimation'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Stage 1 - Estimation] Device: {device}")

    dataset = load_data(image_dir, density_dir)
    print(f"Loaded {len(dataset)} samples")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = VehicleEstimationModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = EstimationTrainer(model, train_loader, val_loader, device, lr)
    trainer.train(num_epochs, save_dir)
