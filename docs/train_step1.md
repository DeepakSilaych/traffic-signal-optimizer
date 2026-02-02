# Step 1: Vehicle Estimation Training

## Overview

Train a model to convert traffic camera images into vehicle density maps.

```
Input:  RGB Image [3, 72, 72]
Output: Density Map [1, 18, 18]
```

The density map is a spatial representation where the sum of all pixel values approximates the total vehicle count.

---

## Model Architecture: REsnext

| Component | Details |
|-----------|---------|
| **Stem** | 2× Conv2d (7×7) + MaxPool for initial feature extraction |
| **ResNeXt Block 1** | Grouped convolution (cardinality=32) with 2× downsampling |
| **ResNeXt Block 2** | Identity block with grouped convolution |
| **Regression Head** | 5×5 conv → 1×1 convs (64→1000→400→1) |
| **Parameters** | ~1.3M trainable parameters |

---

## Data Format

### Option 1: Real Data

```
data/
├── images/
│   ├── 000001.jpg    # 72×72 RGB images
│   ├── 000002.jpg
│   └── ...
└── density_maps/
    ├── 000001.npy    # 18×18 float32 arrays
    ├── 000002.npy
    └── ...
```

### Option 2: Synthetic Data (Default)

Automatically generated for testing. Creates random vehicle patterns with corresponding density maps.

---

## Training

### Quick Start (Synthetic Data)

```bash
cd /path/to/btp
python train/run_all.py --step 1 --epochs 100
```

### With Real Data

```bash
python train/run_all.py --step 1 --data-dir /path/to/data --epochs 100
```

### All Options

```bash
python train/run_all.py --step 1 \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --data-dir /path/to/data \
    --resume /path/to/checkpoint.pt
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 16 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--data-dir` | None | Path to real data (uses synthetic if not set) |
| `--resume` | None | Path to checkpoint to resume from |

---

## Loss Function

Combined loss with two components:

```python
Loss = MSE(pred_density, target_density) + 0.1 × MSE(pred_count, target_count)
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| Pixel MSE | 1.0 | Spatial accuracy of density distribution |
| Count MSE | 0.1 | Total vehicle count accuracy |

---

## Training Output

### Checkpoints

```
checkpoints/estimation/
└── best_model.pt    # Best validation loss model
```

### Checkpoint Contents

```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'best_val_loss': float,
    'metrics': {
        'loss': float,
        'mae': float  # Mean Absolute Error in count
    }
}
```

### Console Output

```
Epoch 0:
  Train - Total: 0.0234, Pixel: 0.0212, Count: 0.0022
  Val   - Loss: 0.0198, MAE: 2.34
Epoch 10:
  Train - Total: 0.0089, Pixel: 0.0078, Count: 0.0011
  Val   - Loss: 0.0076, MAE: 1.12
```

---

## Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Pixel Loss** | MSE between predicted and target density maps | Lower is better |
| **Count Loss** | MSE between predicted and target vehicle counts | Lower is better |
| **MAE** | Mean Absolute Error of vehicle count | < 2.0 vehicles |

---

## Inference

### Single Image

```python
import torch
from models import REsnext

model = REsnext()
checkpoint = torch.load('checkpoints/estimation/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

image = torch.randn(1, 3, 72, 72)  # Your preprocessed image
with torch.no_grad():
    density_map = model(image)
    vehicle_count = density_map.sum().item()

print(f"Estimated vehicles: {vehicle_count:.1f}")
```

### Batch Processing

```python
images = torch.randn(16, 3, 72, 72)
with torch.no_grad():
    density_maps = model(images)
    counts = density_maps.sum(dim=(1, 2, 3))
```

---

## Preparing Real Data

### Image Preprocessing

```python
import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (72, 72))
    img = img.astype(np.float32) / 255.0
    return img
```

### Density Map Generation (from point annotations)

```python
from scipy.ndimage import gaussian_filter

def create_density_map(points, output_size=(18, 18), sigma=1.5):
    density = np.zeros(output_size, dtype=np.float32)
    for x, y in points:
        # Scale coordinates to output size
        x_scaled = int(x * output_size[1] / original_width)
        y_scaled = int(y * output_size[0] / original_height)
        if 0 <= x_scaled < output_size[1] and 0 <= y_scaled < output_size[0]:
            density[y_scaled, x_scaled] += 1
    density = gaussian_filter(density, sigma=sigma)
    return density
```

---

## Hyperparameter Tuning

| Parameter | Range | Notes |
|-----------|-------|-------|
| Learning Rate | 1e-4 to 1e-2 | Start with 1e-3 |
| Batch Size | 8 to 32 | 16 works well |
| Count Weight | 0.05 to 0.2 | Balance spatial vs count accuracy |
| Gaussian Sigma | 1.0 to 2.5 | For density map smoothing |

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Count is accurate but density is blurry | Increase pixel loss weight |
| Training loss not decreasing | Reduce learning rate |
| Overfitting | Add data augmentation, reduce model size |
| Out of memory | Reduce batch size |
