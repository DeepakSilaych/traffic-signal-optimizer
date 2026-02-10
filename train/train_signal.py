import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import torch
import torch.nn as nn

from architecture.signal_optimizer import SignalOptimizer


class SignalTrainer:
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
        total_loss, n = 0, 0
        for past, future, target_durations in self.train_loader:
            past = past.to(self.device)
            future = future.to(self.device)
            target_durations = target_durations.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(past, future)
            loss = nn.functional.mse_loss(out['durations'], target_durations)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item()
            n += 1
        return total_loss / n

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, n = 0, 0
        for past, future, target_durations in self.val_loader:
            past = past.to(self.device)
            future = future.to(self.device)
            target_durations = target_durations.to(self.device)
            out = self.model(past, future)
            loss = nn.functional.mse_loss(out['durations'], target_durations)
            total_loss += loss.item()
            n += 1
        return total_loss / n

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
                    'best_val_loss': self.best_val_loss,
                }, save_path)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")


if __name__ == '__main__':
    print("[Stage 3 - Signal Optimizer] Placeholder - not yet implemented")
    print("Requires dataset of (past_densities, future_densities, target_durations) tuples")
