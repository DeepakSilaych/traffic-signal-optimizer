import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from datetime import datetime
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from architecture.vehicle_estimation_model import VehicleEstimationModel

input_img = 'data/images/0000.jpg'
location = 'jaipur'

model = VehicleEstimationModel()
ckpt = torch.load('checkpoints/estimation/best_model.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

img = Image.open(input_img).convert('RGB').resize((72, 72))
x = torch.FloatTensor(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

with torch.no_grad():
    density = model(x)

d = density[0, 0].numpy()
print(f'Input:  {input_img}')
print(f'Shape:  {x.shape} -> {density.shape}')
print(f'Range:  [{d.min():.4f}, {d.max():.4f}]')
print(f'Count:  {d.sum():.2f}')

output_dir = Path('data/output/estimation')
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = output_dir / f'{location}_{timestamp}.png'

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
im = ax.imshow(d, cmap='jet')
ax.set_title(f'Density (count={d.sum():.1f})')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f'Saved:  {output_path}')
