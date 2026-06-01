"""TEASER++ with FPFH + mutual-NN correspondences."""

import numpy as np

from baselines._common import suppress_stdout, to_o3d, compute_fpfh, mutual_nn_correspondences

NOISE_BOUND = 0.05
VOXEL_SIZE = 0.05
MAX_ITER = 50


def register(source, target):
    import teaserpp_python

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    src_fpfh = compute_fpfh(to_o3d(source), VOXEL_SIZE)
    tgt_fpfh = compute_fpfh(to_o3d(target), VOXEL_SIZE)
    s_idx, t_idx = mutual_nn_correspondences(
        np.asarray(src_fpfh.data).T, np.asarray(tgt_fpfh.data).T)
    if len(s_idx) == 0:
        raise ValueError("FPFH found no mutual correspondences")

    params = teaserpp_python.RobustRegistrationSolver.Params()
    params.noise_bound = NOISE_BOUND
    params.estimate_scaling = False
    params.rotation_max_iterations = MAX_ITER

    solver = teaserpp_python.RobustRegistrationSolver(params)
    with suppress_stdout():
        solver.solve(source[s_idx].T, target[t_idx].T)
    sol = solver.getSolution()
    return np.asarray(sol.rotation), np.asarray(sol.translation)
