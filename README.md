# dualpy

[![CI](https://github.com/hmackk/dualpy/actions/workflows/ci.yml/badge.svg)](https://github.com/hmackk/dualpy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dualpy)](https://pypi.org/project/dualpy/)
[![Python](https://img.shields.io/pypi/pyversions/dualpy)](https://pypi.org/project/dualpy/)
[![License](https://img.shields.io/pypi/l/dualpy)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-hmackk.github.io%2Fdualpy-blue)](https://hmackk.github.io/dualpy/)

Lightweight forward-mode automatic differentiation for NumPy.

Dualpy requires no compilation, no new array API, and installs in under a
second: NumPy is the only dependency. It hooks into NumPy's own dispatch
protocols (`__array_ufunc__` and `__array_function__`) to propagate dual
numbers alongside the primal computation, so your existing functions work
unchanged.

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
grad(np.array([1.0, 1.0]))  # array([0., 0.]), at the minimum
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
| [`jvp`](https://hmackk.github.io/dualpy/api/differentiation/#dualpy.jvp) | `jvp(func, primals, tangents)` | Jacobian-vector product |
| [`jacobian`](https://hmackk.github.io/dualpy/api/differentiation/#dualpy.jacobian) | `jacobian(func, argnums=0)` | Full Jacobian matrix |
| [`derivative`](https://hmackk.github.io/dualpy/api/differentiation/#dualpy.derivative) | `derivative(func, argnums=0)` | Scalar-to-scalar derivative |
| [`nth_derivative`](https://hmackk.github.io/dualpy/api/differentiation/#dualpy.nth_derivative) | `nth_derivative(func, n)` | n-th order derivative via nesting |
| [`gradient`](https://hmackk.github.io/dualpy/api/differentiation/#dualpy.gradient) | `gradient(func, argnums=0)` | Gradient of a scalar-valued function |
| [`hessian`](https://hmackk.github.io/dualpy/api/differentiation/#dualpy.hessian) | `hessian(func, argnums=0)` | Hessian via forward-over-forward |
| [`hvp`](https://hmackk.github.io/dualpy/api/differentiation/#dualpy.hvp) | `hvp(func, v)` | Hessian-vector product in O(n) |
| [`curl`](https://hmackk.github.io/dualpy/api/differentiation/#dualpy.curl) | `curl(func)` | Curl of R^3 -> R^3 |
| [`divergence`](https://hmackk.github.io/dualpy/api/differentiation/#dualpy.divergence) | `divergence(func)` | Divergence of R^n -> R^n |
| [`laplacian`](https://hmackk.github.io/dualpy/api/differentiation/#dualpy.laplacian) | `laplacian(func)` | Laplacian of a scalar field |

See the full [API Reference](https://hmackk.github.io/dualpy/api/differentiation/)
and the complete list of
[supported NumPy operations](https://hmackk.github.io/dualpy/supported-operations/).

## Features

- **Broad NumPy coverage.** Derivative rules for 60+ ufuncs and 100+ array
  routines: elementary arithmetic, trigonometry, exponentials, linear algebra
  (including SVD, QR, Cholesky, eigendecomposition), reductions, shape
  manipulation, and numerical routines.
- **Complex derivatives.** Complex-valued arrays are supported and produce
  correct derivatives for
  [holomorphic functions](https://en.wikipedia.org/wiki/Holomorphic_function).
  Support for [Wirtinger derivatives](https://en.wikipedia.org/wiki/Wirtinger_derivatives)
  (non-holomorphic functions) is planned for a future release.
- **Higher-order derivatives.** Exact n-th order derivatives via nested
  forward-mode (forward-over-forward).
- **Multi-argument differentiation.** Differentiate with respect to any
  positional argument (or several) using `argnums`.
- **Vector calculus.** Built-in differential operators: `curl`, `divergence`,
  and `laplacian`.

## Limitations

- **Forward-mode only.** Most efficient when the number of inputs is small
  relative to the number of outputs. For high-dimensional inputs with scalar
  output (e.g. neural network loss), reverse-mode (backpropagation) is
  faster, consider [JAX](https://github.com/jax-ml/jax) or
  [PyTorch](https://pytorch.org/) for that use case.
- **Non-holomorphic complex functions.** For non-holomorphic operations
  (e.g. `np.abs`, `np.conj` on complex inputs), the tangent propagation does
  not produce Wirtinger derivatives. This is planned for a future release.
- **Registered operations only.** If your code calls a NumPy function that
  dualpy hasn't registered, you'll get `NotImplementedError`.
  The registery mechanism might be exposed to users in future releases
  to add unregistered Numpy and/or their custom functions.
  See the [supported operations](https://hmackk.github.io/dualpy/supported-operations/)
  page for the full list.
- **No GPU support.** Dualpy operates on CPU NumPy arrays.

## License

MIT
