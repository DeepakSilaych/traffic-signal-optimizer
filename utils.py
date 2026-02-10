import torch
import torch.nn as nn


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


class MaskedMAELoss(nn.Module):
    def __init__(self, null_val=0.0):
        super().__init__()
        self.null_val = null_val

    def forward(self, pred, target):
        mask = (target != self.null_val).float()
        mask = mask / (torch.mean(mask) + 1e-8)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(pred - target) * mask
        return torch.mean(torch.where(torch.isnan(loss), torch.zeros_like(loss), loss))


class MaskedMSELoss(nn.Module):
    def __init__(self, null_val=0.0):
        super().__init__()
        self.null_val = null_val

    def forward(self, pred, target):
        mask = (target != self.null_val).float()
        mask = mask / (torch.mean(mask) + 1e-8)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = (pred - target) ** 2 * mask
        return torch.mean(torch.where(torch.isnan(loss), torch.zeros_like(loss), loss))


class MultiHorizonLoss(nn.Module):
    def __init__(self, horizon_weights=None, base_loss='mae'):
        super().__init__()
        self.horizon_weights = horizon_weights
        self.base_loss = MaskedMAELoss() if base_loss == 'mae' else MaskedMSELoss()

    def forward(self, pred, target):
        num_horizons = pred.size(1)
        weights = torch.ones(num_horizons, device=pred.device) / num_horizons if self.horizon_weights is None \
            else torch.tensor(self.horizon_weights, device=pred.device)
        weights = weights / weights.sum()
        horizon_losses = [self.base_loss(pred[:, h], target[:, h]) for h in range(num_horizons)]
        total_loss = sum(w * l for w, l in zip(weights, horizon_losses))
        return total_loss, horizon_losses


class CombinedPredictionLoss(nn.Module):
    def __init__(self, mae_weight=1.0, mse_weight=0.5, temporal_smoothness_weight=0.1):
        super().__init__()
        self.mae_weight = mae_weight
        self.mse_weight = mse_weight
        self.temporal_smoothness_weight = temporal_smoothness_weight
        self.mae_loss = MaskedMAELoss()
        self.mse_loss = MaskedMSELoss()

    def forward(self, pred, target):
        mae = self.mae_loss(pred, target)
        mse = self.mse_loss(pred, target)
        temporal_smoothness = torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2) if pred.size(1) > 1 else 0
        total = self.mae_weight * mae + self.mse_weight * mse + self.temporal_smoothness_weight * temporal_smoothness
        return total, {
            'mae': mae.item(),
            'mse': mse.item(),
            'temporal_smoothness': temporal_smoothness.item() if isinstance(temporal_smoothness, torch.Tensor) else 0
        }


def masked_mae(pred, target, null_val=0.0):
    mask = (target != null_val).float()
    if mask.sum() == 0:
        return torch.tensor(0.0)
    return (torch.abs(pred - target) * mask).sum() / mask.sum()


def masked_mse(pred, target, null_val=0.0):
    mask = (target != null_val).float()
    if mask.sum() == 0:
        return torch.tensor(0.0)
    return ((pred - target) ** 2 * mask).sum() / mask.sum()


def masked_rmse(pred, target, null_val=0.0):
    return torch.sqrt(masked_mse(pred, target, null_val))


def masked_mape(pred, target, null_val=0.0, epsilon=1e-8):
    mask = ((target != null_val) & (target.abs() > epsilon)).float()
    if mask.sum() == 0:
        return torch.tensor(0.0)
    return (torch.abs((pred - target) / (target + epsilon)) * mask).sum() / mask.sum() * 100


def compute_all_metrics(pred, target, null_val=0.0):
    return {
        'mae': masked_mae(pred, target, null_val).item(),
        'mse': masked_mse(pred, target, null_val).item(),
        'rmse': masked_rmse(pred, target, null_val).item(),
        'mape': masked_mape(pred, target, null_val).item()
    }


def compute_horizon_metrics(pred, target, prediction_horizons, null_val=0.0):
    return {
        f'horizon_{h}': compute_all_metrics(pred[:, i], target[:, i], null_val)
        for i, h in enumerate(prediction_horizons)
    }
