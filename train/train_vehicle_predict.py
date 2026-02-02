import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models import VehiclePredictor, VehiclePredictorLite
from data import create_synthetic_dataset, VehiclePredictionDataset, PrecomputedDensityDataset
from utils import MultiHorizonLoss, compute_horizon_metrics


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
        
        self.criterion = MultiHorizonLoss(base_loss='mae')
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
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions, _ = self.model(inputs, mode='multi_horizon')
            
            loss, horizon_losses = self.criterion(predictions, targets)
            
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
        num_batches = 0
        all_preds = []
        all_targets = []
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            predictions, _ = self.model(inputs, mode='multi_horizon')
            loss, _ = self.criterion(predictions, targets)
            
            total_loss += loss.item()
            num_batches += 1
            all_preds.append(predictions.cpu())
            all_targets.append(targets.cpu())
            
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = compute_horizon_metrics(
            all_preds, all_targets, self.prediction_horizons
        )
        
        return total_loss / num_batches, metrics
    
    def train(self, num_epochs, save_path=None):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_metrics)
                    
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
                for horizon, metrics in val_metrics.items():
                    print(f"  {horizon}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")
                    
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, path, epoch, metrics):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics
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
            in_channels=config.get('in_channels', 1),
            num_zones=config.get('num_zones', 8),
            embed_dim=config.get('embed_dim', 32),
            spatial_size=config.get('spatial_size', 18),
            hidden_dim=config.get('hidden_dim', 64),
            num_layers=config.get('num_layers', 2),
            prediction_horizons=config.get('prediction_horizons', [1, 3, 5])
        )
    else:
        return VehiclePredictor(
            in_channels=config.get('in_channels', 1),
            num_zones=config.get('num_zones', 8),
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
            prediction_horizons=config.get('prediction_horizons', [1, 3, 5, 10, 15]),
            num_zones=config.get('num_zones', 8)
        )
    else:
        dataset = create_synthetic_dataset(
            num_samples=config.get('num_samples', 2000),
            spatial_size=config.get('spatial_size', 18),
            num_zones=config.get('num_zones', 8),
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
    save_dir.mkdir(parents=True, exist_ok=True)
    
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
        start_epoch = trainer.load_checkpoint(config['resume_from'])
        print(f"Resumed from epoch {start_epoch}")
    
    trainer.train(
        num_epochs=config.get('num_epochs', 100),
        save_path=save_dir / 'best_model.pt'
    )
    
    return trainer


if __name__ == '__main__':
    config = {
        'in_channels': 1,
        'num_zones': 8,
        'embed_dim': 64,
        'spatial_size': 18,
        'num_encoder_layers': 4,
        'num_heads': 4,
        'mlp_ratio': 4,
        'dropout': 0.1,
        'prediction_horizons': [1, 3, 5, 10, 15],
        'max_seq_len': 50,
        'seq_len': 24,
        'lite': False,
        'num_samples': 2000,
        'batch_size': 16,
        'num_workers': 0,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'save_dir': 'checkpoints/prediction',
        'density_dir': None,
        'resume_from': None
    }
    
    run_training(config)
