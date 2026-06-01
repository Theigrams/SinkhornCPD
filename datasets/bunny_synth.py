"""Bunny synthetic data: load + augment + pair generation.

Convention:
    source = Y = clean CAD/template (downsampled bunny)
    target = X = source after crop → noise → outliers → rigid transform
    R_gt, t_gt are the GT rigid transform applied to make target.
"""

from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


SYNTH_DIR = Path(__file__).parent / "synth"


def load_ply(path):
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def normalize(points):
    """Center on bbox midpoint and scale to unit ball."""
    points = points[~np.isnan(points).any(1)]
    points = points - (points.max(0) + points.min(0)) / 2
    return points / np.max(np.linalg.norm(points, axis=1))


def voxel_downsample(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return np.asarray(pcd.voxel_down_sample(voxel_size).points)


def random_downsample(points, n, seed=0):
    if len(points) <= n:
        return points
    return points[np.random.default_rng(seed).choice(len(points), n, replace=False)]


def add_noise(points, sigma, rng):
    return points if sigma <= 0 else points + rng.standard_normal(points.shape) * sigma


def add_outliers(points, ratio, rng, scale=2.0):
    """Replace `ratio` fraction of points with random samples from a unit ball ×scale."""
    if ratio <= 0:
        return points
    n_out = int(ratio * len(points))
    if n_out <= 0:
        return points
    idx = rng.choice(len(points), n_out, replace=False)
    dirs = rng.standard_normal((n_out, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = rng.random(n_out) ** (1 / 3)
    out = (dirs * radii[:, None]).astype(np.float32) * scale  # legacy: see datasets/synth/ provenance
    points = points.copy()
    points[idx] = out
    return points


def crop_with_plane(points, keep_ratio, rng):
    """Cut along a random plane, keeping the top `keep_ratio` fraction by signed distance."""
    if keep_ratio >= 1.0:
        return points
    n_keep = int(len(points) * keep_ratio + 0.5)
    normal = rng.standard_normal(3)
    normal /= np.linalg.norm(normal)
    return points[np.argsort(-(points @ normal))[:n_keep]]


def random_rotation(angle_deg, rng, axis=None):
    if axis is None:
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(np.deg2rad(angle_deg) * np.asarray(axis)).as_matrix()


def random_translation(rng, low=-0.5, high=0.5):
    return rng.uniform(low, high, 3)


def make_pair(source, *, sigma, ratio, overlap, angle, seed):
    """Build a single (target, R_gt, t_gt) following the paper protocol."""
    rng = np.random.default_rng(seed)
    R = random_rotation(angle, rng)
    t = random_translation(rng)
    target = source.copy()
    if overlap < 1.0:
        target = crop_with_plane(target, overlap, rng)
    target = add_noise(target, sigma, rng)
    target = add_outliers(target, ratio, rng)
    target = target @ R.T + t
    return target, R, t


def load_synth(axis, level):
    """Load a pre-generated benchmark slice.

    Returns (source, targets, R_gts, t_gts) where targets is an object array of
    variable-length point clouds (because crop changes point count).
    """
    fname = {
        "noise":   f"sigma_{level:.3f}.npz",
        "outlier": f"ratio_{level:.3f}.npz",
        "overlap": f"overlap_{level:.3f}.npz",
        "rotation": f"angle_{int(level):03d}.npz",
    }[axis]
    source = np.load(SYNTH_DIR / "source.npy")
    blob = np.load(SYNTH_DIR / axis / fname, allow_pickle=True)
    return source, blob["target"], blob["R_gt"], blob["t_gt"]


# ============================================================================
# Offline benchmark generator. Reproduces datasets/synth/{source.npy, ...}.
#
# Usage:
#     python -m datasets.bunny_synth                       # default: write to datasets/synth/
#     python -m datasets.bunny_synth --output-dir datasets/synth_check  # for diffing
# ============================================================================

if __name__ == "__main__":
    import argparse

    DEFAULTS = dict(sigma=0.02, ratio=0.2, overlap=0.9, angle=30.0)
    AXES = {
        "noise":    ("sigma",   [0.01, 0.02, 0.03, 0.04, 0.05]),
        "outlier":  ("ratio",   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        "overlap":  ("overlap", [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
        "rotation": ("angle",   [10, 20, 30, 40, 50, 60, 70, 80, 90]),
    }

    parser = argparse.ArgumentParser(description="Regenerate Bunny synthetic registration benchmark.")
    parser.add_argument("--source-ply", type=Path, default=Path(__file__).parent / "bunny.ply")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "synth")
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--num-points", type=int, default=3000)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.source_ply}")
    src_full = voxel_downsample(load_ply(args.source_ply), args.voxel_size)
    source = random_downsample(src_full, args.num_points, seed=0)
    print(f"  source: full(after voxel)={len(src_full)} -> downsampled={len(source)}")
    np.save(args.output_dir / "source.npy", source)

    for axis, (param_name, levels) in AXES.items():
        out_dir = args.output_dir / axis
        out_dir.mkdir(parents=True, exist_ok=True)
        for val in levels:
            kwargs = dict(DEFAULTS)
            kwargs[param_name] = val

            targets, Rs, ts = [], [], []
            for i in range(args.trials):
                tgt, R, t = make_pair(source, **kwargs, seed=i)
                targets.append(tgt)
                Rs.append(R)
                ts.append(t)

            fname = (
                f"{param_name}_{val:.3f}.npz" if isinstance(val, float)
                else f"{param_name}_{int(val):03d}.npz"
            )
            np.savez(
                out_dir / fname,
                target=np.array(targets, dtype=object),
                R_gt=np.array(Rs),
                t_gt=np.array(ts),
            )
            print(f"  {axis}/{fname}")

    print("Done.")
