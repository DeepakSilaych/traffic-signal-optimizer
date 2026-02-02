import torch
import numpy as np


def masked_mae(pred, target, null_val=0.0):
    mask = target != null_val
    mask = mask.float()
    
    if mask.sum() == 0:
        return torch.tensor(0.0)
        
    mae = torch.abs(pred - target)
    mae = mae * mask
    
    return mae.sum() / mask.sum()


def masked_mse(pred, target, null_val=0.0):
    mask = target != null_val
    mask = mask.float()
    
    if mask.sum() == 0:
        return torch.tensor(0.0)
        
    mse = (pred - target) ** 2
    mse = mse * mask
    
    return mse.sum() / mask.sum()


def masked_rmse(pred, target, null_val=0.0):
    return torch.sqrt(masked_mse(pred, target, null_val))


def masked_mape(pred, target, null_val=0.0, epsilon=1e-8):
    mask = (target != null_val) & (target.abs() > epsilon)
    mask = mask.float()
    
    if mask.sum() == 0:
        return torch.tensor(0.0)
        
    mape = torch.abs((pred - target) / (target + epsilon))
    mape = mape * mask
    
    return (mape.sum() / mask.sum()) * 100


def compute_all_metrics(pred, target, null_val=0.0):
    return {
        'mae': masked_mae(pred, target, null_val).item(),
        'mse': masked_mse(pred, target, null_val).item(),
        'rmse': masked_rmse(pred, target, null_val).item(),
        'mape': masked_mape(pred, target, null_val).item()
    }


def compute_horizon_metrics(pred, target, prediction_horizons, null_val=0.0):
    metrics_per_horizon = {}
    
    for h_idx, horizon in enumerate(prediction_horizons):
        h_pred = pred[:, h_idx]
        h_target = target[:, h_idx]
        
        metrics_per_horizon[f'horizon_{horizon}'] = compute_all_metrics(
            h_pred, h_target, null_val
        )
        
    return metrics_per_horizon
