import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.ndimage import gaussian_filter

from architecture.flow_prediction import ST3DNet

HORIZONS = {
    '15min': {'steps': 1, 'save': 'checkpoints/prediction/st3dnet_15min.pt'},
    '30min': {'steps': 2, 'save': 'checkpoints/prediction/st3dnet_30min.pt'},
    '1hr':   {'steps': 4, 'save': 'checkpoints/prediction/st3dnet_1hr.pt'},
}


def create_dummy_prediction_data(num_samples=200, height=18, width=18, T_c=6, T_w=4, horizon_steps=1):
    closeness = np.zeros((num_samples, 1, T_c, height, width), dtype=np.float32)
    weekly = np.zeros((num_samples, 1, T_w, height, width), dtype=np.float32)
    targets = np.zeros((num_samples, 1, height, width), dtype=np.float32)

    t = np.linspace(0, 50, num_samples)
    base = np.sin(t * 0.1) * 0.3 + 0.5

    hotspots = [
        (height // 4, width // 4),
        (height // 4, 3 * width // 4),
        (3 * height // 4, width // 4),
        (3 * height // 4, 3 * width // 4),
    ]

    for i in range(num_samples):
        for step in range(T_c):
            dm = np.zeros((height, width), dtype=np.float32)
            for cy, cx in hotspots:
                y, x = np.ogrid[:height, :width]
                dm += np.exp(-((x - cx)**2 + (y - cy)**2) / 18.0) * (base[i] + step * 0.02)
            dm += np.random.rand(height, width) * 0.05
            closeness[i, 0, step] = gaussian_filter(np.clip(dm, 0, 1), sigma=1.0)

        for step in range(T_w):
            dm = np.zeros((height, width), dtype=np.float32)
            for cy, cx in hotspots:
                y, x = np.ogrid[:height, :width]
                dm += np.exp(-((x - cx)**2 + (y - cy)**2) / 18.0) * base[i]
            weekly[i, 0, step] = gaussian_filter(np.clip(dm, 0, 1), sigma=1.0)

        target = np.zeros((height, width), dtype=np.float32)
        offset = horizon_steps * 0.05
        for cy, cx in hotspots:
            y, x = np.ogrid[:height, :width]
            target += np.exp(-((x - cx)**2 + (y - cy)**2) / 18.0) * (base[i] + offset)
        targets[i, 0] = gaussian_filter(np.clip(target, 0, 1), sigma=1.0)

    return torch.FloatTensor(closeness), torch.FloatTensor(weekly), torch.FloatTensor(targets)


class PredictionTrainer:
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
        total_loss, n = 0, 0
        for x_c, x_w, target in self.train_loader:
            x_c, x_w, target = x_c.to(self.device), x_w.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x_c, x_w)
            loss = self.criterion(pred, target)
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
        for x_c, x_w, target in self.val_loader:
            x_c, x_w, target = x_c.to(self.device), x_w.to(self.device), target.to(self.device)
            pred = self.model(x_c, x_w)
            loss = self.criterion(pred, target)
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
                print(f"  Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}")


if __name__ == '__main__':
    T_c = 6
    T_w = 4
    height = 18
    width = 18
    num_samples = 200
    num_epochs = 20
    batch_size = 16
    lr = 1e-3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for name, cfg in HORIZONS.items():
        print(f"\n[Stage 2 - ST-3DNet {name}] Device: {device}")

        x_c, x_w, targets = create_dummy_prediction_data(
            num_samples, height, width, T_c, T_w, horizon_steps=cfg['steps']
        )
        dataset = TensorDataset(x_c, x_w, targets)
        print(f"  Samples: {len(dataset)}, Horizon: {name} ({cfg['steps']} steps ahead)")

        train_size = int(0.8 * len(dataset))
        train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = ST3DNet(in_channels=1, T_c=T_c, T_w=T_w, height=height, width=width)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        PredictionTrainer(model, train_loader, val_loader, device, lr).train(num_epochs, cfg['save'])
