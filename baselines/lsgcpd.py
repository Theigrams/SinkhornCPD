"""LSG-CPD via MATLAB Engine.  Engine is started lazily and kept alive."""

import os

import numpy as np

W = 0.5
MAX_ITER = 50

_ENGINE = None
_LSG_DIR = os.path.join(os.path.dirname(__file__), "..", "third_party", "LSG-CPD")


def _engine():
    global _ENGINE
    if _ENGINE is None:
        import matlab.engine

        _ENGINE = matlab.engine.start_matlab()
        _ENGINE.addpath(os.path.abspath(_LSG_DIR), nargout=0)
    return _ENGINE


def register(source, target):
    import matlab

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    eng = _engine()
    R, t = eng.run_lsgcpd_engine(
        matlab.double(source.tolist()),
        matlab.double(target.tolist()),
        W,
        MAX_ITER,
        nargout=2,
    )
    return np.asarray(R), np.asarray(t).flatten()


def close():
    global _ENGINE
    if _ENGINE is not None:
        _ENGINE.quit()
        _ENGINE = None
