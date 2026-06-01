"""Registration baselines.  Each module exposes:  register(source, target) -> (R, t).

Convention: source=Y=CAD, target=X=scan; aligned = source @ R.T + t ≈ target.
Timing is the experiment layer's job — wrappers do not measure time.
"""

from . import cpd, fgr, filterreg, rpot

try:
    from . import robot
except ImportError:  # geomloss not installed
    robot = None

# C++ / MATLAB backends — wrap import so missing third_party doesn't break the package
try:
    from . import teaser
except ImportError:  # teaserpp_python not installed
    teaser = None

try:
    from . import sparseicp
except ImportError:
    sparseicp = None

try:
    from . import lsgcpd
except ImportError:  # matlab.engine not installed
    lsgcpd = None


def _maybe(mod):
    return mod.register if mod is not None else None


METHODS = {
    "CPD": cpd.register,
    "FilterReg": filterreg.register,
    "FGR": fgr.register,
    "RPOT": rpot.register,
    "RobOT": _maybe(robot),
    "TEASER++": _maybe(teaser),
    "Sparse-ICP": _maybe(sparseicp),
    "LSG-CPD": _maybe(lsgcpd),
}
