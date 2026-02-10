import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict

from architecture.vehicle_estimation_model import VehicleEstimationModel
from architecture.flow_prediction import ST3DNet


ESTIMATION_CKPT = 'checkpoints/estimation/best_model.pt'
PREDICTION_CKPTS = {
    '15min': 'checkpoints/prediction/st3dnet_15min.pt',
    '30min': 'checkpoints/prediction/st3dnet_30min.pt',
    '1hr':   'checkpoints/prediction/st3dnet_1hr.pt',
}
T_C = 6
T_W = 4
HISTORY_DIR = 'data/density_history'


def _load_model(cls, ckpt_path, **kwargs):
    model = cls(**kwargs)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


class Pipeline:
    def __init__(self, location='default', device='cpu'):
        self.location = location
        self.device = device
        self.history_dir = Path(HISTORY_DIR) / location
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.estimation_model = _load_model(
            VehicleEstimationModel, ESTIMATION_CKPT
        ).to(device)

        self.prediction_models = {
            name: _load_model(
                ST3DNet, path, in_channels=1, T_c=T_C, T_w=T_W, height=18, width=18
            ).to(device)
            for name, path in PREDICTION_CKPTS.items()
        }

    def _density_path(self, ts: datetime) -> Path:
        return self.history_dir / f"{ts.strftime('%Y%m%d_%H%M%S')}.npy"

    def _save_density(self, density: np.ndarray, ts: datetime):
        np.save(str(self._density_path(ts)), density)

    def _load_closeness(self, ts: datetime) -> Optional[torch.Tensor]:
        interval = timedelta(minutes=15)
        maps = []
        for i in range(T_C, 0, -1):
            t = ts - i * interval
            path = self._find_nearest(t, tolerance_minutes=10)
            if path is None:
                return None
            maps.append(np.load(str(path)))
        stacked = np.stack(maps, axis=0)
        return torch.FloatTensor(stacked).unsqueeze(0).unsqueeze(0)

    def _load_weekly(self, ts: datetime) -> Optional[torch.Tensor]:
        maps = []
        for w in range(T_W, 0, -1):
            t = ts - timedelta(weeks=w)
            path = self._find_nearest(t, tolerance_minutes=30)
            if path is None:
                return None
            maps.append(np.load(str(path)))
        stacked = np.stack(maps, axis=0)
        return torch.FloatTensor(stacked).unsqueeze(0).unsqueeze(0)

    def _find_nearest(self, target: datetime, tolerance_minutes=10) -> Optional[Path]:
        best_path, best_diff = None, timedelta(minutes=tolerance_minutes + 1)
        for f in self.history_dir.glob('*.npy'):
            try:
                file_ts = datetime.strptime(f.stem, '%Y%m%d_%H%M%S')
            except ValueError:
                continue
            diff = abs(file_ts - target)
            if diff < best_diff:
                best_diff = diff
                best_path = f
        return best_path

    @torch.no_grad()
    def estimate(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert('RGB').resize((72, 72))
        x = torch.FloatTensor(np.array(img, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return self.estimation_model(x).cpu().numpy()[0, 0]

    @torch.no_grad()
    def predict(self, x_c: torch.Tensor, x_w: torch.Tensor) -> Dict[str, np.ndarray]:
        results = {}
        for name, model in self.prediction_models.items():
            pred = model(x_c.to(self.device), x_w.to(self.device))
            results[name] = pred.cpu().numpy()[0, 0]
        return results

    def run(self, image_path: str, timestamp: Optional[datetime] = None) -> Dict:
        ts = timestamp or datetime.now()

        density = self.estimate(image_path)
        self._save_density(density, ts)

        result = {
            'timestamp': ts,
            'density': density,
            'count': float(density.sum()),
            'predictions': None,
        }

        x_c = self._load_closeness(ts)
        x_w = self._load_weekly(ts)

        if x_c is not None and x_w is not None:
            result['predictions'] = self.predict(x_c, x_w)
        else:
            missing = []
            if x_c is None:
                missing.append(f'closeness (need {T_C} recent maps)')
            if x_w is None:
                missing.append(f'weekly (need {T_W} weekly maps)')
            result['missing'] = missing

        return result
