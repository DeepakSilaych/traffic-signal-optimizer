# Step 2: Vehicle Prediction Training

## Overview

Train a model to predict future vehicle counts from a sequence of density maps.

```
Input:  Density Sequence [T=24, 1, 18, 18]  (24 minutes of history)
Output: Future Counts [H=5, N=8]            (5 horizons × 8 zones)
```

Prediction horizons: 1, 3, 5, 10, 15 minutes into the future.

---

## Model Architecture: VehiclePredictor

### Full Model (Transformer-based)

```
Density Sequence [B, 24, 1, 18, 18]
         │
         ▼
┌─────────────────────────────────┐
│  ZoneAttentionPooling           │
│  - CNN spatial embedding        │
│  - Cross-attention to 8 zones   │
│  Output: [B, 24, 8, 64]         │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  SpatioTemporalEncoder          │
│  - 4 transformer layers         │
│  - Alternating spatial/temporal │
│  - Positional encoding          │
│  Output: [B, 24, 8, 64]         │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  MultiHorizonHead               │
│  - 5 separate MLPs              │
│  - One per prediction horizon   │
│  Output: [B, 5, 8]              │
└─────────────────────────────────┘
```

### Lite Model (LSTM-based)

```
Density Sequence [B, 24, 1, 18, 18]
         │
         ▼
┌─────────────────────────────────┐
│  ZoneAttentionPooling           │
│  Output: [B, 24, 8, 32]         │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  2-layer LSTM                   │
│  Hidden dim: 64                 │
│  Output: [B, 64]                │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Linear Heads (per horizon)     │
│  Output: [B, 5, 8]              │
└─────────────────────────────────┘
```

| Model | Parameters | Use Case |
|-------|------------|----------|
| VehiclePredictor | ~500K | High accuracy, server deployment |
| VehiclePredictorLite | ~100K | Fast inference, edge devices |

---

## Data Format

### Option 1: Precomputed Density Maps (from Step 1)

```
density_maps/
├── 000001.npy    # Sequential density maps (18×18)
├── 000002.npy    # Named in temporal order
├── 000003.npy
└── ...
```

### Option 2: Synthetic Data (Default)

Automatically generated temporal patterns with realistic traffic flow characteristics.

---

## Training

### Quick Start (Synthetic Data)

```bash
cd /path/to/btp
python train/run_all.py --step 2 --epochs 100
```

### With Precomputed Density Maps

```bash
python train/run_all.py --step 2 --data-dir /path/to/density_maps --epochs 100
```

### Lite Model (Faster Training)

```bash
python train/run_all.py --step 2 --lite --epochs 100
```

### All Options

```bash
python train/run_all.py --step 2 \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --data-dir /path/to/density_maps \
    --resume /path/to/checkpoint.pt \
    --lite
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 16 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--data-dir` | None | Path to density maps (uses synthetic if not set) |
| `--resume` | None | Path to checkpoint to resume from |
| `--lite` | False | Use lightweight LSTM model |

---

## Loss Function: MultiHorizonLoss

```python
Loss = (1/H) × Σ MAE(pred[h], target[h])
```

Masked MAE loss that handles missing values (null_val=0).

| Component | Description |
|-----------|-------------|
| Horizon Weights | Equal weighting across all prediction horizons |
| Base Loss | Mean Absolute Error (MAE) |
| Masking | Ignores zero values in target |

---

## Training Output

### Checkpoints

```
checkpoints/prediction/
└── best_model.pt    # Best validation loss model
```

### Checkpoint Contents

```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'best_val_loss': float,
    'metrics': {
        'horizon_1': {'mae': float, 'rmse': float, 'mape': float},
        'horizon_3': {'mae': float, 'rmse': float, 'mape': float},
        'horizon_5': {'mae': float, 'rmse': float, 'mape': float},
        'horizon_10': {'mae': float, 'rmse': float, 'mape': float},
        'horizon_15': {'mae': float, 'rmse': float, 'mape': float}
    }
}
```

### Console Output

```
Epoch 0: Train Loss=0.4532, Val Loss=0.4123
  horizon_1: MAE=0.3421, RMSE=0.4532
  horizon_3: MAE=0.3876, RMSE=0.5012
  horizon_5: MAE=0.4234, RMSE=0.5543
  horizon_10: MAE=0.4987, RMSE=0.6234
  horizon_15: MAE=0.5432, RMSE=0.6876
```

---

## Metrics Per Horizon

| Metric | Description | Target |
|--------|-------------|--------|
| **MAE** | Mean Absolute Error | Lower is better |
| **RMSE** | Root Mean Square Error | Lower is better |
| **MAPE** | Mean Absolute Percentage Error | < 15% |

Expected pattern: Error increases with prediction horizon.

---

## Configuration

### Model Configuration

```python
config = {
    'in_channels': 1,           # Density map channels
    'num_zones': 8,             # Number of attention zones
    'embed_dim': 64,            # Embedding dimension
    'spatial_size': 18,         # Density map size
    'num_encoder_layers': 4,    # Transformer layers
    'num_heads': 4,             # Attention heads
    'mlp_ratio': 4,             # MLP expansion ratio
    'dropout': 0.1,             # Dropout rate
    'prediction_horizons': [1, 3, 5, 10, 15],  # Minutes ahead
    'seq_len': 24,              # Input sequence length (minutes)
}
```

### Lite Model Configuration

```python
config = {
    'in_channels': 1,
    'num_zones': 8,
    'embed_dim': 32,            # Smaller embedding
    'spatial_size': 18,
    'hidden_dim': 64,           # LSTM hidden size
    'num_layers': 2,            # LSTM layers
    'prediction_horizons': [1, 3, 5],  # Fewer horizons
    'lite': True
}
```

---

## Inference

### Multi-Horizon Prediction

```python
import torch
from models import VehiclePredictor

model = VehiclePredictor(
    num_zones=8,
    prediction_horizons=[1, 3, 5, 10, 15]
)
checkpoint = torch.load('checkpoints/prediction/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Input: 24 density maps (24 minutes of history)
density_sequence = torch.randn(1, 24, 1, 18, 18)

with torch.no_grad():
    predictions, zone_attention = model(density_sequence, mode='multi_horizon')

# predictions shape: [1, 5, 8] = [batch, horizons, zones]
print(f"1-min prediction: {predictions[0, 0]}")   # 8 zone counts
print(f"15-min prediction: {predictions[0, 4]}")  # 8 zone counts
```

### Autoregressive Prediction (Variable Horizon)

```python
with torch.no_grad():
    predictions, _ = model(density_sequence, mode='autoregressive', num_steps=20)

# predictions shape: [1, 20, 8] = 20 future timesteps
```

### Visualize Zone Attention

```python
import matplotlib.pyplot as plt

# zone_attention shape: [B, T, num_zones, H*W]
attn = zone_attention[0, -1]  # Last timestep attention
attn = attn.view(8, 18, 18)   # Reshape to spatial

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(attn[i].numpy(), cmap='hot')
    ax.set_title(f'Zone {i+1}')
plt.savefig('zone_attention.png')
```

---

## Pipeline: Step 1 → Step 2

### Online Inference

```python
from models import InferencePipeline

pipeline = InferencePipeline(
    estimation_checkpoint='checkpoints/estimation/best_model.pt',
    prediction_checkpoint='checkpoints/prediction/best_model.pt',
    prediction_config={
        'num_zones': 8,
        'prediction_horizons': [1, 3, 5, 10, 15]
    }
)

# Process frames one by one
for frame in video_frames:
    result = pipeline.process_frame(frame)
    
    print(f"Current count: {result['vehicle_count']:.1f}")
    
    if 'predictions' in result:  # After 24 frames
        print(f"1-min forecast: {result['predictions'][0, 0].sum():.1f}")
        print(f"15-min forecast: {result['predictions'][0, 4].sum():.1f}")
```

### Batch Processing

```python
from models import TrafficSignalPipeline

pipeline = TrafficSignalPipeline(
    estimation_model=estimation_model,
    prediction_model=prediction_model,
    freeze_estimation=True
)

# Input: sequence of images
images = torch.randn(1, 24, 3, 72, 72)
output = pipeline(images, mode='multi_horizon')

density_maps = output['density_maps']    # [1, 24, 1, 18, 18]
predictions = output['predictions']      # [1, 5, 8]
zone_attention = output['zone_attention']
```

---

## Hyperparameter Tuning

| Parameter | Range | Notes |
|-----------|-------|-------|
| Learning Rate | 1e-4 to 5e-3 | Start with 1e-3 |
| Batch Size | 8 to 32 | Limited by sequence memory |
| Sequence Length | 12 to 48 | Longer = more context, more memory |
| Number of Zones | 4 to 16 | Match intersection complexity |
| Encoder Layers | 2 to 6 | More layers = more capacity |
| Embed Dimension | 32 to 128 | Trade-off accuracy vs speed |

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Long-horizon predictions poor | Increase sequence length, add more training data |
| Overfitting | Add dropout, reduce model size, data augmentation |
| Training unstable | Reduce learning rate, add gradient clipping |
| Out of memory | Reduce batch size, use lite model |
| Zone attention unfocused | Train longer, check data quality |

---

## Zone Interpretation

The 8 learned zones typically correspond to:

| Zone | Typical Meaning |
|------|-----------------|
| 1-2 | North approach lanes |
| 3-4 | South approach lanes |
| 5-6 | East approach lanes |
| 7-8 | West approach lanes |

The attention mechanism automatically learns which spatial regions are important for each zone. Visualize attention maps to verify zone assignments match intersection geometry.
