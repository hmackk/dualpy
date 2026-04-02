# dualpy

[![CI](https://github.com/hmackk/dualpy/actions/workflows/ci.yml/badge.svg)](https://github.com/hmackk/dualpy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dualpy)](https://pypi.org/project/dualpy/)
[![Python](https://img.shields.io/pypi/pyversions/dualpy)](https://pypi.org/project/dualpy/)
[![License](https://img.shields.io/pypi/l/dualpy)](https://github.com/hmackk/dualpy/blob/main/LICENSE)

**Lightweight forward-mode automatic differentiation for NumPy.**

Dualpy lets you compute exact derivatives of ordinary NumPy code. No
compilation step, no new array API, and nothing heavy to install: NumPy is
the only dependency.

```bash
pip install dualpy
```

## At a glance

Write a function with standard NumPy operations and differentiate it
automatically:

```python
import numpy as np
from dualpy import gradient

def loss(params):
    """Predict y = params @ x and return MSE loss."""
    x = np.array([1.0, 2.0, 3.0])
    y_true = np.array([2.0, 4.0, 6.0])
    residuals = params @ np.vstack([x, np.ones_like(x)]) - y_true
    return np.mean(residuals ** 2)

grad = gradient(loss)
grad(np.array([2.0, 0.0]))  # exact gradient at the initial guess
```

Dualpy hooks into NumPy's dispatch protocols (`__array_ufunc__` and
`__array_function__`) to propagate [dual numbers](getting-started.md#how-it-works)
alongside the primal computation, so your existing functions work unchanged.

## What's covered

Derivative rules for **60+ ufuncs** and **100+ array routines**: elementary
arithmetic, trigonometry, exponentials, linear algebra (including matrix
decompositions like SVD, QR, Cholesky, and eigendecomposition), reductions,
shape manipulation, and numerical routines. See the full list on the
[Supported Operations](supported-operations.md) page.

## Features

- **Complex derivatives.** Complex-valued arrays are supported and produce
  correct derivatives for
  [holomorphic functions](https://en.wikipedia.org/wiki/Holomorphic_function).
- **Higher-order derivatives.** Exact \(n\)-th order derivatives via nested
  forward-mode (forward-over-forward).
- **Multi-argument differentiation.** Differentiate with respect to any
  positional argument (or several) using `argnums`.
- **Vector calculus.** Built-in differential operators: `curl`, `divergence`,
  and `laplacian`.

## Limitations

- **Forward-mode only.** Most efficient when the number of inputs is small
  relative to the number of outputs. For high-dimensional inputs with scalar
  output (e.g. neural network loss), reverse-mode (backpropagation) is
  faster: consider [JAX](https://github.com/jax-ml/jax) or
  [PyTorch](https://pytorch.org/) for that use case.
- **Registered operations only.** If your code calls a NumPy function that
  dualpy hasn't registered, you'll get `NotImplementedError`. The set of
  registered operations covers common scientific computing needs, and it will
  be expanded further.
- **No GPU support.** Dualpy operates on CPU NumPy arrays.

## Next steps

- [Getting Started](getting-started.md): installation, quickstart examples,
  and how dual numbers work under the hood.
- [API Reference](api/differentiation.md): full signatures and docstrings
  for every public function.
- [Supported Operations](supported-operations.md): every NumPy operation
  dualpy can differentiate through.
