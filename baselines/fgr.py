"""FGR via Open3D (FPFH + RANSAC-free fast matching)."""

import numpy as np
import open3d as o3d

from baselines._common import compute_fpfh, to_o3d

VOXEL_SIZE = 0.05
MAX_ITER = 50


def register(source, target):
    src = to_o3d(source)
    tgt = to_o3d(target)
    # FGR uses lighter normal/FPFH params than TEASER (matches paper)
    src_fpfh = compute_fpfh(src, VOXEL_SIZE, normal_mult=2, fpfh_mult=5, normal_nn=30, fpfh_nn=100)
    tgt_fpfh = compute_fpfh(tgt, VOXEL_SIZE, normal_mult=2, fpfh_mult=5, normal_nn=30, fpfh_nn=100)
    out = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src,
        tgt,
        src_fpfh,
        tgt_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=VOXEL_SIZE * 1.5,
            iteration_number=MAX_ITER,
        ),
    )
    T = out.transformation
    return np.asarray(T[:3, :3]), np.asarray(T[:3, 3])
