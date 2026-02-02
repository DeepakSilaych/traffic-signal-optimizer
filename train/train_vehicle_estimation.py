import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models import REsnext
from data import create_synthetic_estimation_data, VehicleEstimationFromFiles


class DensityMapLoss(nn.Module):
    def __init__(self, count_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.count_weight = count_weight
        
    def forward(self, pred, target):
        pixel_loss = self.mse(pred, target)
        
        pred_count = pred.sum(dim=(1, 2, 3))
        target_count = target.sum(dim=(1, 2, 3))
        count_loss = torch.mean((pred_count - target_count) ** 2)
        
        return pixel_loss + self.count_weight * count_loss, {
            'pixel_loss': pixel_loss.item(),
            'count_loss': count_loss.item()
        }


class VehicleEstimationTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        lr=1e-3,
        weight_decay=1e-5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = DensityMapLoss(count_weight=0.1)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_pixel_loss = 0
        total_count_loss = 0
        num_batches = 0
        
        for images, density_maps in self.train_loader:
            images = images.to(self.device)
            density_maps = density_maps.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(images)
            
            loss, loss_dict = self.criterion(predictions, density_maps)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_pixel_loss += loss_dict['pixel_loss']
            total_count_loss += loss_dict['count_loss']
            num_batches += 1
            
        return {
            'total': total_loss / num_batches,
            'pixel': total_pixel_loss / num_batches,
            'count': total_count_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        for images, density_maps in self.val_loader:
            images = images.to(self.device)
            density_maps = density_maps.to(self.device)
            
            predictions = self.model(images)
            loss, _ = self.criterion(predictions, density_maps)
            
            pred_count = predictions.sum(dim=(1, 2, 3))
            target_count = density_maps.sum(dim=(1, 2, 3))
            mae = torch.mean(torch.abs(pred_count - target_count))
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
            
        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches
        }
    
    def train(self, num_epochs, save_dir=None):
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            self.scheduler.step()
            
            self.train_losses.append(train_metrics['total'])
            self.val_losses.append(val_metrics['loss'])
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                if save_dir:
                    self.save_checkpoint(save_dir / 'best_model.pt', epoch, val_metrics)
                    
            if epoch % 10 == 0:
                print(f"Epoch {epoch}:")
                print(f"  Train - Total: {train_metrics['total']:.4f}, "
                      f"Pixel: {train_metrics['pixel']:.4f}, Count: {train_metrics['count']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}")
                
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, path, epoch, metrics):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics
        }, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        return checkpoint.get('epoch', 0)


def run_training(config):
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if config.get('data_dir'):
        dataset = VehicleEstimationFromFiles(
            image_dir=config['data_dir'] / 'images',
            density_dir=config['data_dir'] / 'density_maps'
        )
    else:
        dataset = create_synthetic_estimation_data(
            num_samples=config.get('num_samples', 2000),
            image_size=72,
            density_size=18,
            max_vehicles=config.get('max_vehicles', 50)
        )
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    model = REsnext()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = VehicleEstimationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    if config.get('resume_from'):
        start_epoch = trainer.load_checkpoint(config['resume_from'])
        print(f"Resumed from epoch {start_epoch}")
        
    trainer.train(
        num_epochs=config.get('num_epochs', 100),
        save_dir=config.get('save_dir', 'checkpoints/estimation')
    )
    
    return trainer


if __name__ == '__main__':
    config = {
        'num_samples': 2000,
        'max_vehicles': 50,
        'batch_size': 16,
        'num_workers': 0,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'save_dir': 'checkpoints/estimation',
        'data_dir': None,
        'resume_from': None
    }
    
    run_training(config)
