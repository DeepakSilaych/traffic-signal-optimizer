import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from architecture.flow_prediction import ST3DNet

location = 'jaipur'

HORIZONS = {
    '15min': 'checkpoints/prediction/st3dnet_15min.pt',
    '30min': 'checkpoints/prediction/st3dnet_30min.pt',
    '1hr':   'checkpoints/prediction/st3dnet_1hr.pt',
}

x_c = torch.randn(1, 1, 6, 18, 18)
x_w = torch.randn(1, 1, 4, 18, 18)

output_dir = Path('data/output/prediction')
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

fig, axes = plt.subplots(1, len(HORIZONS), figsize=(5 * len(HORIZONS), 4))

for idx, (name, ckpt_path) in enumerate(HORIZONS.items()):
    model = ST3DNet(in_channels=1, T_c=6, T_w=4, height=18, width=18)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with torch.no_grad():
        pred = model(x_c, x_w)

    d = pred[0, 0].numpy()
    print(f'{name}: range=[{d.min():.4f}, {d.max():.4f}], count={d.sum():.2f}')

    im = axes[idx].imshow(d, cmap='jet')
    axes[idx].set_title(f'{name} (count={d.sum():.1f})')
    axes[idx].axis('off')
    plt.colorbar(im, ax=axes[idx], fraction=0.046)

plt.tight_layout()
output_path = output_dir / f'{location}_{timestamp}.png'
plt.savefig(output_path, dpi=150)
print(f'Saved: {output_path}')
