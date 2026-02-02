import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMAELoss(nn.Module):
    def __init__(self, null_val=0.0):
        super().__init__()
        self.null_val = null_val
        
    def forward(self, pred, target):
        mask = target != self.null_val
        mask = mask.float()
        mask /= torch.mean(mask) + 1e-8
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        
        loss = torch.abs(pred - target)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        
        return torch.mean(loss)


class MaskedMSELoss(nn.Module):
    def __init__(self, null_val=0.0):
        super().__init__()
        self.null_val = null_val
        
    def forward(self, pred, target):
        mask = target != self.null_val
        mask = mask.float()
        mask /= torch.mean(mask) + 1e-8
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        
        loss = (pred - target) ** 2
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        
        return torch.mean(loss)


class MultiHorizonLoss(nn.Module):
    def __init__(self, horizon_weights=None, base_loss='mae'):
        super().__init__()
        self.horizon_weights = horizon_weights
        self.base_loss = MaskedMAELoss() if base_loss == 'mae' else MaskedMSELoss()
        
    def forward(self, pred, target):
        num_horizons = pred.size(1)
        
        if self.horizon_weights is None:
            weights = torch.ones(num_horizons, device=pred.device) / num_horizons
        else:
            weights = torch.tensor(self.horizon_weights, device=pred.device)
            weights = weights / weights.sum()
            
        total_loss = 0
        horizon_losses = []
        
        for h in range(num_horizons):
            h_loss = self.base_loss(pred[:, h], target[:, h])
            horizon_losses.append(h_loss)
            total_loss += weights[h] * h_loss
            
        return total_loss, horizon_losses


class CombinedPredictionLoss(nn.Module):
    def __init__(
        self,
        mae_weight=1.0,
        mse_weight=0.5,
        temporal_smoothness_weight=0.1
    ):
        super().__init__()
        self.mae_weight = mae_weight
        self.mse_weight = mse_weight
        self.temporal_smoothness_weight = temporal_smoothness_weight
        
        self.mae_loss = MaskedMAELoss()
        self.mse_loss = MaskedMSELoss()
        
    def forward(self, pred, target):
        mae = self.mae_loss(pred, target)
        mse = self.mse_loss(pred, target)
        
        temporal_smoothness = 0
        if pred.size(1) > 1:
            diff = pred[:, 1:] - pred[:, :-1]
            temporal_smoothness = torch.mean(diff ** 2)
            
        total_loss = (
            self.mae_weight * mae + 
            self.mse_weight * mse + 
            self.temporal_smoothness_weight * temporal_smoothness
        )
        
        return total_loss, {
            'mae': mae.item(),
            'mse': mse.item(),
            'temporal_smoothness': temporal_smoothness.item() if isinstance(temporal_smoothness, torch.Tensor) else temporal_smoothness
        }
