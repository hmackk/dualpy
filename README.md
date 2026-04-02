# dualpy

[![CI](https://github.com/hmackk/dualpy/actions/workflows/ci.yml/badge.svg)](https://github.com/hmackk/dualpy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dualpy)](https://pypi.org/project/dualpy/)
[![Python](https://img.shields.io/pypi/pyversions/dualpy)](https://pypi.org/project/dualpy/)
[![License](https://img.shields.io/pypi/l/dualpy)](LICENSE)

Lightweight forward-mode automatic differentiation for NumPy.

Dualpy lets you compute exact derivatives of ordinary NumPy code. It hooks
into NumPy's own dispatch protocols (`__array_ufunc__` and
`__array_function__`) to propagate dual numbers alongside the primal
computation, so your existing functions work unchanged. NumPy is the only
dependency — no compilation step, no new array API to learn, and nothing
heavy to install.

Dualpy ships derivative rules for a broad set of NumPy operations — elementary
arithmetic, trigonometry, exponentials, linear algebra (including matrix
decompositions like SVD, QR, Cholesky, and eigendecomposition), reductions,
shape manipulation, and numerical routines — covering the needs of scientific
research, engineering, and education. Unregistered operations raise
`NotImplementedError`, more operations will be added in future releases,
but for the most part, most are there.

## Installation

```bash
pip install dualpy
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add dualpy
```

## Quickstart

### Gradient

```python
import numpy as np
from dualpy import gradient

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

grad = gradient(rosenbrock)
grad(np.array([1.0, 1.0]))  # array([0., 0.])  — at the minimum
```

### Jacobian

```python
from dualpy import jacobian

def f(x):
    return np.stack([x[0]**2, x[0] * x[1]])

jacobian(f)(np.array([2.0, 3.0]))
# array([[4., 0.],
#        [3., 2.]])
```

### Hessian

```python
from dualpy import hessian

def quadratic(x):
    return x[0]**2 + 3 * x[1]**2

hessian(quadratic)(np.array([1.0, 1.0]))
# array([[2., 0.],
#        [0., 6.]])
```

### Low-level JVP

```python
from dualpy import jvp

f = lambda x: np.sin(x) * np.exp(x)
primal, tangent = jvp(f, np.array(1.0), np.array(1.0))
# tangent is df/dx at x=1
```

## API

| Function | Signature | Description |
|---|---|---|
| `jvp` | `jvp(func, primals, tangents)` | Jacobian-vector product |
| `jacobian` | `jacobian(func, argnums=0)` | Full Jacobian matrix |
| `derivative` | `derivative(func, argnums=0)` | Scalar-to-scalar derivative |
| `nth_derivative` | `nth_derivative(func, n)` | n-th order derivative via nesting |
| `gradient` | `gradient(func, argnums=0)` | Gradient of a scalar-valued function |
| `hessian` | `hessian(func, argnums=0)` | Hessian via forward-over-forward |
| `hvp` | `hvp(func, v)` | Hessian-vector product in O(n) |
| `curl` | `curl(func)` | Curl of R^3 -> R^3 |
| `divergence` | `divergence(func)` | Divergence of R^n -> R^n |
| `laplacian` | `laplacian(func)` | Laplacian of a scalar field |

## Features

- **Complex derivatives.** Complex-valued arrays are supported and produce
  correct derivatives for
  [holomorphic functions](https://en.wikipedia.org/wiki/Holomorphic_function).
  Support for [Wirtinger derivatives](https://en.wikipedia.org/wiki/Wirtinger_derivatives)
  (non-holomorphic functions) is planned for a future release.
- **Higher-order derivatives.** Exact n-th order derivatives via nested
  forward-mode (forward-over-forward).
- **Multi-argument differentiation.** Differentiate with respect to any
  positional argument (or several) using `argnums`.

## Limitations

- **Forward-mode only.** Most efficient when the number of inputs is small
  relative to the number of outputs. For high-dimensional inputs with scalar
  output (e.g. neural network loss), reverse-mode (backpropagation) is
  faster: consider JAX or PyTorch for that use case.
- **Non-holomorphic complex functions.** For non-holomorphic operations
  (e.g. `np.abs`, `np.conj` on complex inputs), the tangent propagation does
  not produce Wirtinger derivatives. This is planned for a future release.
- **Registered operations only.** If your code calls a NumPy function that
  dualpy hasn't registered, you'll get `NotImplementedError`. The set of
  registered operations covers common scientific computing needs, and it will be expanded further.
- **No GPU support.** Dualpy operates on CPU NumPy arrays.

## License

MIT
