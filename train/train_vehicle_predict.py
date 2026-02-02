import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models import VehiclePredictor, VehiclePredictorLite
from data import create_synthetic_dataset, PrecomputedDensityDataset


class DensityMapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        pixel_loss = self.mse(pred, target)
        
        pred_counts = pred.sum(dim=(2, 3, 4))
        target_counts = target.sum(dim=(2, 3, 4))
        count_loss = self.mse(pred_counts, target_counts)
        
        return pixel_loss + 0.1 * count_loss


class VehiclePredictorTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        lr=1e-3,
        weight_decay=1e-5,
        prediction_horizons=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.prediction_horizons = prediction_horizons or [1, 3, 5, 10, 15]
        
        self.criterion = DensityMapLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(inputs)
            loss = self.criterion(predictions, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            predictions = self.model(inputs)
            loss = self.criterion(predictions, targets)
            
            pred_counts = predictions.sum(dim=(2, 3, 4))
            target_counts = targets.sum(dim=(2, 3, 4))
            mae = torch.abs(pred_counts - target_counts).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
            
        return total_loss / num_batches, total_mae / num_batches
    
    def train(self, num_epochs, save_path=None):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_mae = self.validate()
            
            self.scheduler.step()
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if save_path:
                    self.save_checkpoint(save_path, epoch)
                    
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, MAE={val_mae:.2f}")
                
    def save_checkpoint(self, path, epoch):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        return checkpoint.get('epoch', 0)


def create_model(config):
    if config.get('lite', False):
        return VehiclePredictorLite(
            embed_dim=config.get('embed_dim', 32),
            spatial_size=config.get('spatial_size', 18),
            hidden_dim=config.get('hidden_dim', 64),
            num_layers=config.get('num_layers', 2),
            prediction_horizons=config.get('prediction_horizons', [1, 3, 5])
        )
    else:
        return VehiclePredictor(
            embed_dim=config.get('embed_dim', 64),
            spatial_size=config.get('spatial_size', 18),
            num_encoder_layers=config.get('num_encoder_layers', 4),
            num_heads=config.get('num_heads', 4),
            mlp_ratio=config.get('mlp_ratio', 4),
            dropout=config.get('dropout', 0.1),
            prediction_horizons=config.get('prediction_horizons', [1, 3, 5, 10, 15]),
            max_seq_len=config.get('max_seq_len', 50)
        )


def run_training(config):
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if config.get('density_dir'):
        dataset = PrecomputedDensityDataset(
            density_dir=config['density_dir'],
            seq_len=config.get('seq_len', 24),
            prediction_horizons=config.get('prediction_horizons', [1, 3, 5, 10, 15])
        )
    else:
        dataset = create_synthetic_dataset(
            num_samples=config.get('num_samples', 2000),
            spatial_size=config.get('spatial_size', 18),
            seq_len=config.get('seq_len', 24),
            prediction_horizons=config.get('prediction_horizons', [1, 3, 5, 10, 15])
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
    
    model = create_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    save_dir = Path(config.get('save_dir', 'checkpoints/prediction'))
    
    trainer = VehiclePredictorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5),
        prediction_horizons=config.get('prediction_horizons', [1, 3, 5, 10, 15])
    )
    
    if config.get('resume_from'):
        trainer.load_checkpoint(config['resume_from'])
    
    trainer.train(
        num_epochs=config.get('num_epochs', 100),
        save_path=save_dir / 'best_model.pt'
    )
    
    return trainer


if __name__ == '__main__':
    config = {
        'embed_dim': 64,
        'spatial_size': 18,
        'num_encoder_layers': 4,
        'num_heads': 4,
        'prediction_horizons': [1, 3, 5, 10, 15],
        'seq_len': 24,
        'lite': False,
        'num_samples': 2000,
        'batch_size': 16,
        'lr': 1e-3,
        'num_epochs': 100,
        'save_dir': 'checkpoints/prediction'
    }
    
    run_training(config)
