import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models import TMAE, SMAE, STDMAEForDensityMap
from data import create_synthetic_dataset, PrecomputedDensityDataset


class PretrainTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=1e-3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for inputs, _ in self.train_loader:
            inputs = inputs.to(self.device)
            B, T, C, H, W = inputs.shape
            inputs = inputs.view(B, T, -1, C).permute(0, 1, 2, 3)
            
            self.optimizer.zero_grad()
            loss, _ = self.model(inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        for inputs, _ in self.val_loader:
            inputs = inputs.to(self.device)
            B, T, C, H, W = inputs.shape
            inputs = inputs.view(B, T, -1, C).permute(0, 1, 2, 3)
            
            loss, _ = self.model(inputs)
            total_loss += loss.item()
            
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'best_val_loss': self.best_val_loss
                }, save_path)
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")


class DownstreamTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=1e-3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
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
            
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        
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
            
        n = len(self.val_loader)
        return total_loss / n, total_mae / n
    
    def train(self, num_epochs, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_mae = self.validate()
            self.scheduler.step()
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'best_val_loss': self.best_val_loss
                }, save_path)
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, MAE={val_mae:.2f}")


def pretrain_tmae(config):
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[T-MAE Pre-training] Device: {device}")
    
    dataset = create_synthetic_dataset(
        num_samples=config.get('num_samples', 2000),
        spatial_size=config.get('spatial_size', 18),
        seq_len=config.get('seq_len', 24),
        prediction_horizons=config.get('prediction_horizons', [1, 3, 5, 10, 15])
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 16), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 16), shuffle=False)
    
    num_nodes = config.get('spatial_size', 18) ** 2
    model = TMAE(
        num_nodes=num_nodes,
        in_channels=1,
        patch_size=config.get('patch_size', 12),
        embed_dim=config.get('embed_dim', 96),
        encoder_depth=config.get('encoder_depth', 4),
        decoder_depth=config.get('decoder_depth', 1),
        num_heads=config.get('num_heads', 4),
        mask_ratio=config.get('mask_ratio', 0.75)
    )
    print(f"T-MAE parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = PretrainTrainer(model, train_loader, val_loader, device, lr=config.get('lr', 1e-3))
    trainer.train(config.get('pretrain_epochs', 50), 'checkpoints/prediction/tmae_pretrained.pt')


def pretrain_smae(config):
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[S-MAE Pre-training] Device: {device}")
    
    dataset = create_synthetic_dataset(
        num_samples=config.get('num_samples', 2000),
        spatial_size=config.get('spatial_size', 18),
        seq_len=config.get('seq_len', 24),
        prediction_horizons=config.get('prediction_horizons', [1, 3, 5, 10, 15])
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 16), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 16), shuffle=False)
    
    num_nodes = config.get('spatial_size', 18) ** 2
    model = SMAE(
        num_nodes=num_nodes,
        in_channels=1,
        embed_dim=config.get('embed_dim', 96),
        encoder_depth=config.get('encoder_depth', 4),
        decoder_depth=config.get('decoder_depth', 1),
        num_heads=config.get('num_heads', 4),
        mask_ratio=config.get('mask_ratio', 0.75)
    )
    print(f"S-MAE parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = PretrainTrainer(model, train_loader, val_loader, device, lr=config.get('lr', 1e-3))
    trainer.train(config.get('pretrain_epochs', 50), 'checkpoints/prediction/smae_pretrained.pt')


def train_downstream(config):
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Downstream Training] Device: {device}")
    
    dataset = create_synthetic_dataset(
        num_samples=config.get('num_samples', 2000),
        spatial_size=config.get('spatial_size', 18),
        seq_len=config.get('seq_len', 24),
        prediction_horizons=config.get('prediction_horizons', [1, 3, 5, 10, 15])
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 16), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 16), shuffle=False)
    
    model = STDMAEForDensityMap(
        spatial_size=config.get('spatial_size', 18),
        in_channels=1,
        patch_size=config.get('patch_size', 12),
        embed_dim=config.get('embed_dim', 96),
        encoder_depth=config.get('encoder_depth', 4),
        decoder_depth=config.get('decoder_depth', 1),
        num_heads=config.get('num_heads', 4),
        hidden_dim=config.get('hidden_dim', 256),
        prediction_horizons=config.get('prediction_horizons', [1, 3, 5, 10, 15])
    )
    
    tmae_path = config.get('tmae_checkpoint', 'checkpoints/prediction/tmae_pretrained.pt')
    smae_path = config.get('smae_checkpoint', 'checkpoints/prediction/smae_pretrained.pt')
    
    if Path(tmae_path).exists() and Path(smae_path).exists():
        print("Loading pre-trained encoders...")
        model.load_pretrained(tmae_path, smae_path)
        if config.get('freeze_encoders', True):
            model.freeze_encoders()
            print("Encoders frozen")
    else:
        print("No pre-trained weights found, training from scratch")
        
    print(f"STD-MAE parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    trainer = DownstreamTrainer(model, train_loader, val_loader, device, lr=config.get('lr', 1e-3))
    trainer.train(config.get('num_epochs', 100), 'checkpoints/prediction/stdmae_best.pt')


def run_training(config):
    print("=" * 60)
    print("Phase 1: Pre-training T-MAE")
    print("=" * 60)
    pretrain_tmae(config)
    
    print("\n" + "=" * 60)
    print("Phase 2: Pre-training S-MAE")
    print("=" * 60)
    pretrain_smae(config)
    
    print("\n" + "=" * 60)
    print("Phase 3: Training Downstream Predictor")
    print("=" * 60)
    train_downstream(config)


if __name__ == '__main__':
    config = {
        'spatial_size': 18,
        'patch_size': 12,
        'embed_dim': 96,
        'encoder_depth': 4,
        'decoder_depth': 1,
        'num_heads': 4,
        'hidden_dim': 256,
        'mask_ratio': 0.75,
        'prediction_horizons': [1, 3, 5, 10, 15],
        'seq_len': 24,
        'num_samples': 2000,
        'batch_size': 16,
        'lr': 1e-3,
        'pretrain_epochs': 50,
        'num_epochs': 100,
        'freeze_encoders': True
    }
    
    run_training(config)
