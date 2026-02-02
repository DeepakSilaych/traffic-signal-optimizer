# Step 2: Vehicle Prediction Training (STD-MAE)

## Overview

Train a model to predict future density maps from a sequence of past density maps using **Spatial-Temporal-Decoupled Masked Pre-training (STD-MAE)**.

```
Input:  [B, 24, 1, 18, 18]   # 24 past density maps (24 min history)
Output: [B, 5, 1, 18, 18]    # 5 future density maps (at horizons 1,3,5,10,15 min)
```

Reference: [STD-MAE Paper (IJCAI-24)](https://arxiv.org/abs/2312.00516)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STD-MAE ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: T-MAE PRE-TRAINING          PHASE 2: S-MAE PRE-TRAINING          │
│  ════════════════════════════          ════════════════════════════         │
│                                                                             │
│  Long Input [T_long, N, C]             Long Input [T_long, N, C]            │
│         │                                     │                             │
│         ▼                                     ▼                             │
│  ┌─────────────────┐                   ┌─────────────────┐                  │
│  │ Patch Embedding │                   │Spatial Embedding│                  │
│  │ S-T Positional  │                   │ S-T Positional  │                  │
│  │ Temporal Mask   │◄─ 75% masked      │ Spatial Mask    │◄─ 75% masked     │
│  │ Transformer Enc │                   │ Transformer Enc │                  │
│  └────────┬────────┘                   └────────┬────────┘                  │
│           │                                     │                           │
│           ▼                                     ▼                           │
│  Temporal Representation               Spatial Representation              │
│           │                                     │                           │
│           ▼                                     ▼                           │
│  ┌─────────────────┐                   ┌─────────────────┐                  │
│  │ Temporal Decoder│                   │ Spatial Decoder │                  │
│  │ Reconstruct time│                   │ Reconstruct space│                 │
│  └─────────────────┘                   └─────────────────┘                  │
│                                                                             │
│  PHASE 3: DOWNSTREAM PREDICTION                                            │
│  ══════════════════════════════                                            │
│                                                                             │
│  Short Input [T, N, C]                                                     │
│         │                                                                   │
│         ├──────────────────┬──────────────────┬─────────────────┐          │
│         ▼                  ▼                  ▼                 │          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │          │
│  │ T-MAE Enc   │    │ S-MAE Enc   │    │ Downstream  │         │          │
│  │ (frozen)    │    │ (frozen)    │    │ S-T Predictor│        │          │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │          │
│         │                  │                  │                 │          │
│         ▼                  ▼                  ▼                 │          │
│      T-Repr             S-Repr            ST-Repr              │          │
│         │                  │                  │                 │          │
│         ▼                  ▼                  ▼                 │          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │          │
│  │   MLP       │    │    MLP      │    │    MLP      │         │          │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │          │
│         │                  │                  │                 │          │
│         └────────┬─────────┴──────────────────┘                 │          │
│                  ▼                                              │          │
│           ┌───────────┐                                         │          │
│           │  Fusion   │  (concatenate 3 sources)                │          │
│           │  [3×hidden]│                                        │          │
│           └─────┬─────┘                                         │          │
│                 │                                               │          │
│                 ▼                                               │          │
│         ┌─────────────┐                                         │          │
│         │  FC Layers  │                                         │          │
│         │ (per horizon)│                                        │          │
│         └──────┬──────┘                                         │          │
│                │                                                │          │
│                ▼                                                │          │
│  Future Density Maps [B, 5, 1, 18, 18]                         │          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Components

| Component | File | Purpose |
|-----------|------|---------|
| **T-MAE** | `models/tmae.py` | Temporal Masked Autoencoder - learns temporal patterns |
| **S-MAE** | `models/smae.py` | Spatial Masked Autoencoder - learns spatial patterns |
| **STD-MAE** | `models/stdmae.py` | Combined model with downstream predictor |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spatial_size` | 18 | Density map size (18×18) |
| `patch_size` | 12 | Temporal patch size for T-MAE |
| `embed_dim` | 96 | Embedding dimension |
| `encoder_depth` | 4 | Number of transformer layers |
| `decoder_depth` | 1 | Decoder layers (lightweight) |
| `num_heads` | 4 | Attention heads |
| `mask_ratio` | 0.75 | Masking ratio during pre-training |
| `hidden_dim` | 256 | Downstream predictor hidden size |

---

## Training

### Full Training Pipeline (3 Phases)

```bash
python train/run_all.py --step 2 --pretrain-epochs 50 --epochs 100
```

This runs:
1. **Phase 1**: T-MAE pre-training (50 epochs)
2. **Phase 2**: S-MAE pre-training (50 epochs)  
3. **Phase 3**: Downstream predictor training (100 epochs)

### Command Options

| Flag | Default | Description |
|------|---------|-------------|
| `--step 2` | required | Run step 2 training |
| `--epochs` | 100 | Downstream training epochs |
| `--pretrain-epochs` | 50 | Pre-training epochs for T-MAE and S-MAE |
| `--batch-size` | 16 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--data-dir` | None | Path to density maps (uses synthetic if not set) |

---

## Checkpoints

```
checkpoints/prediction/
├── tmae_pretrained.pt    # Pre-trained T-MAE encoder
├── smae_pretrained.pt    # Pre-trained S-MAE encoder
└── stdmae_best.pt        # Final downstream model
```

---

## Data Format

### Input: Sequence of Density Maps

```
density_maps/
├── 000001.npy    # Shape: [18, 18], dtype: float32
├── 000002.npy    # Sequential in time
├── 000003.npy
└── ...
```

### Synthetic Data (Default)

For testing, synthetic data is auto-generated with:
- Realistic traffic flow patterns (sinusoidal base)
- 4 spatial hotspots simulating intersection approaches
- Gaussian smoothing for natural density distribution

---

## Inference

```python
from inference import TrafficPredictor

predictor = TrafficPredictor(
    estimation_checkpoint='checkpoints/estimation/best_model.pt',
    prediction_checkpoint='checkpoints/prediction/stdmae_best.pt'
)

# Process frames
for frame in video_frames:
    result = predictor.process_frame(frame)
    
    print(f"Current: {result['current_count']:.0f} vehicles")
    
    if result['buffer_ready']:
        # future_densities: [5, 1, 18, 18] - full spatial maps
        # future_counts: [5] - total counts per horizon
        print(f"1-min forecast: {result['future_counts'][0]:.0f}")
        print(f"5-min forecast: {result['future_counts'][2]:.0f}")
        print(f"15-min forecast: {result['future_counts'][4]:.0f}")
```

---

## Why STD-MAE?

| Benefit | Description |
|---------|-------------|
| **Decoupled Learning** | Separate encoders for spatial and temporal patterns |
| **Self-supervised Pre-training** | Learns rich representations from unlabeled data |
| **Masked Reconstruction** | Forces model to understand underlying patterns |
| **Transfer Learning** | Pre-trained encoders can be reused across datasets |

---

## Loss Functions

### Pre-training (T-MAE / S-MAE)
```
Loss = MSE(reconstructed, original)
```
Reconstruction loss on masked patches.

### Downstream
```
Loss = MSE(predicted_density, target_density)
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error on density maps |
| **MAE** | Mean Absolute Error on vehicle counts |

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Pre-training loss not decreasing | Reduce mask_ratio to 0.5 |
| OOM during training | Reduce batch_size or embed_dim |
| Poor downstream performance | Increase pretrain_epochs |
| Spatial patterns not captured | Check S-MAE reconstruction quality |
