# datasets/

Source meshes, pre-generated benchmarks, and dataset loaders.

```
datasets/
├── bunny.ply               # source mesh for Bunny synth
├── bunny_synth.py          # Bunny loader + augment ops + offline generator
├── synth/                  # Bunny synth pre-generated benchmark
│   ├── source.npy              # (3000, 3) downsampled source
│   ├── noise/sigma_*.npz       # σ ∈ {0.01, 0.02, 0.03, 0.04, 0.05}
│   ├── outlier/ratio_*.npz     # ratio ∈ {0.1, …, 0.7}
│   ├── overlap/overlap_*.npz   # overlap ∈ {0.4, …, 0.9}
│   └── rotation/angle_*.npz    # angle ∈ {10°, …, 90°}
└── modelnet/data.npz       # ModelNet40 test pairs (2148 × 2048 points)
```

Each Bunny `.npz` holds 20 trials: `target` (object array of (Nᵢ, 3)),
`R_gt` (20, 3, 3), `t_gt` (20, 3).
All datasets follow `target ≈ source @ R_gt.T + t_gt`.

## Bunny — `bunny_synth.py`

`source.npy` = `bunny.ply` → `voxel_downsample(0.04)` → `random_downsample(3000, seed=0)`.

Each pair (`make_pair`, seed = 0..19):

1. `R = random_rotation(angle, rng)`, `t = random_translation(rng)` (uniform in (−0.5, 0.5)³)
2. `crop_with_plane(target, overlap, rng)` — random plane, keep top fraction
3. `add_noise(target, sigma, rng)` — `N(0, σ²I)` per point
4. `add_outliers(target, ratio, rng, scale=2.0)` — replace `ratio·N` points uniformly in unit ball ×2.0
5. `target = target @ R.T + t`

Defaults for non-swept axes: σ=0.02, ratio=0.2, overlap=0.9, angle=30°.

Regenerate (bit-exact):

```bash
python -m datasets.bunny_synth
```

## ModelNet40 — `modelnet/data.npz`

ModelNet40 test split through GeoTransformer's `ModelNetPairDataset` with deterministic seeding. **GeoTransformer is not vendored**; install it from
<https://github.com/qinzheng93/GeoTransformer> and prepare its ModelNet data per its README. Then run:

```python
import numpy as np
from tqdm import tqdm
from geotransformer.datasets.registration.modelnet.dataset import ModelNetPairDataset

dataset = ModelNetPairDataset(
    dataset_root="<GeoTransformer>/data/ModelNet",
    subset="test",
    num_points=2048,             # uniform downsample per shape
    voxel_size=None,
    rotation_magnitude=30.0,     # max rotation (deg)
    translation_magnitude=0.5,
    noise_magnitude=0.05,        # per-point Gaussian σ
    keep_ratio=0.9,              # asymmetric planar crop
    crop_method="plane",
    asymmetric=True,             # crop ref only, leave src intact
    class_indices="all",
    deterministic=True,          # per-sample RNG by index → reproducible
    twice_sample=True,           # independent downsample for src/ref
    twice_transform=False,
)
n = len(dataset)
sources, targets = [], []
R_gts, t_gts = np.zeros((n, 3, 3)), np.zeros((n, 3))
for i in tqdm(range(n)):
    s = dataset[i]
    sources.append(s["src_points"].astype(np.float64))
    targets.append(s["ref_points"].astype(np.float64))
    R_gts[i], t_gts[i] = s["transform"][:3, :3], s["transform"][:3, 3]
np.savez("datasets/modelnet/data.npz",
         source=np.array(sources, dtype=object),
         target=np.array(targets, dtype=object),
         R_gt=R_gts, t_gt=t_gts)
```

`data.npz` keys: `source`, `target` (object arrays of (2048, 3)), `R_gt` (2148, 3, 3), `t_gt` (2148, 3).
