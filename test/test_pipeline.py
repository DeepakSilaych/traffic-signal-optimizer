import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from inference import Pipeline

location = 'jaipur'
input_img = 'data/images/0000.jpg'

pipeline = Pipeline(location=location)

now = datetime(2026, 2, 11, 8, 0, 0)

print("--- Seeding weekly history (same hour, 4 prior weeks) ---")
for w in range(4, 0, -1):
    ts = now - timedelta(weeks=w)
    r = pipeline.run(input_img, timestamp=ts)
    print(f"  {ts} | count={r['count']:.1f}")

print("\n--- Seeding recent closeness history (last 6 intervals) ---")
for i in range(6, 0, -1):
    ts = now - timedelta(minutes=15 * i)
    r = pipeline.run(input_img, timestamp=ts)
    print(f"  {ts} | count={r['count']:.1f}")

print("\n--- Running full pipeline on current frame ---")
result = pipeline.run(input_img, timestamp=now)
print(f"  Timestamp: {result['timestamp']}")
print(f"  Current count: {result['count']:.1f}")

if result['predictions']:
    print(f"  Predictions available:")
    for name, pred in result['predictions'].items():
        print(f"    {name}: count={pred.sum():.1f}, range=[{pred.min():.4f}, {pred.max():.4f}]")

    output_dir = Path('data/output/pipeline')
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_str = now.strftime('%Y%m%d_%H%M%S')

    horizons = result['predictions']
    ncols = 1 + len(horizons)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    im = axes[0].imshow(result['density'], cmap='jet')
    axes[0].set_title(f"Current (count={result['count']:.1f})")
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    for idx, (name, pred) in enumerate(horizons.items(), 1):
        im = axes[idx].imshow(pred, cmap='jet')
        axes[idx].set_title(f"{name} (count={pred.sum():.1f})")
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046)

    plt.tight_layout()
    out_path = output_dir / f'{location}_{ts_str}.png'
    plt.savefig(out_path, dpi=150)
    print(f"\n  Saved: {out_path}")
else:
    print(f"  Cannot predict yet: {result.get('missing')}")
