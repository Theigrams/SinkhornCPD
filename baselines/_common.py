"""Helpers shared across feature-based baselines (FGR / TEASER++)."""

import os
from contextlib import contextmanager

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@contextmanager
def suppress_stdout():
    """Mute C++ libraries that print to fd 1."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def to_o3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def compute_fpfh(pcd, voxel_size, normal_mult=3, fpfh_mult=6, normal_nn=50, fpfh_nn=150):
    """Estimate normals + FPFH descriptors at the given voxel scale."""
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * normal_mult, max_nn=normal_nn))
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * fpfh_mult, max_nn=fpfh_nn)
    )


def mutual_nn_correspondences(src_feats, tgt_feats):
    """Return index pairs that are mutual nearest neighbors in feature space."""
    tgt_tree = cKDTree(tgt_feats)
    src_tree = cKDTree(src_feats)
    nn_s2t = tgt_tree.query(src_feats, k=1, workers=-1)[1]
    nn_t2s = src_tree.query(tgt_feats, k=1, workers=-1)[1]
    src_idx = np.arange(len(nn_s2t))
    keep = nn_t2s[nn_s2t] == src_idx
    return src_idx[keep], nn_s2t[keep]
