"""CPD via probreg (GPU)."""

import cupy as cp
import numpy as np
from probreg import cpd

W = 0.5
MAX_ITER = 50
TOL = 1e-4


def register(source, target):
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    out = cpd.registration_cpd(
        source,
        target,
        tf_type_name="rigid",
        w=W,
        maxiter=MAX_ITER,
        tol=TOL,
        update_scale=True,
        use_cuda=True,
    )
    # in our experiments, we find that using update_scale=True is better than False
    R = cp.asnumpy(out.transformation.rot)
    t = cp.asnumpy(out.transformation.t)
    return R, t
