from importlib.metadata import version

from . import routines as routines  # noqa: F401
from . import ufuncs as ufuncs  # noqa: F401
from .differentiation import (
    curl,
    derivative,
    divergence,
    gradient,
    hessian,
    hvp,
    jacobian,
    jvp,
    laplacian,
    nth_derivative,
)

__version__ = version("dualpy")

__all__ = [
    "__version__",
    "curl",
    "derivative",
    "divergence",
    "gradient",
    "hessian",
    "hvp",
    "jacobian",
    "jvp",
    "laplacian",
    "nth_derivative",
]
