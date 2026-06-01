"""Sparse-ICP via the prebuilt C++ binary in third_party/sparseicp/build/sparseicp.

Communicates by writing OBJ files to a tempdir; the binary writes back the
registered point cloud, from which we recover (R, t) by Procrustes.
"""

import os
import subprocess
import tempfile

import numpy as np

EXE = os.path.join(os.path.dirname(__file__), "..", "third_party", "sparseicp", "build", "sparseicp")


def _write_obj(points, path):
    with open(path, "w") as f:
        for p in points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")


def _read_obj(path):
    pts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                pts.append([float(x), float(y), float(z)])
    return np.array(pts)


def _procrustes(src, dst):
    cs = src.mean(0)
    cd = dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, cd - R @ cs


def register(source, target):
    if not os.path.exists(EXE):
        raise FileNotFoundError(f"Sparse-ICP binary missing: {EXE}")
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    with tempfile.TemporaryDirectory() as d:
        src_p = os.path.join(d, "source.obj")
        tgt_p = os.path.join(d, "target.obj")
        out_p = os.path.join(d, "out.obj")
        _write_obj(source, src_p)
        _write_obj(target, tgt_p)
        res = subprocess.run([EXE, src_p, tgt_p, out_p], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"sparseicp binary failed: {res.stderr}")
        registered = _read_obj(out_p)
    return _procrustes(source, registered)
