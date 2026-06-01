"""FilterReg via probreg."""

import numpy as np
from probreg import filterreg

W = 0.1
MAX_ITER = 50
TOL = 1e-4


def register(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    out = filterreg.registration_filterreg(
        source,
        target,
        w=W,
        maxiter=MAX_ITER,
        tol=TOL,
        update_sigma2=True,
    )
    R = np.asarray(out.transformation.rot)
    t = np.asarray(out.transformation.t)
    return R, t
