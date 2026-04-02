# dualpy

[![CI](https://github.com/hmackk/dualpy/actions/workflows/ci.yml/badge.svg)](https://github.com/hmackk/dualpy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dualpy)](https://pypi.org/project/dualpy/)
[![Python](https://img.shields.io/pypi/pyversions/dualpy)](https://pypi.org/project/dualpy/)
[![License](https://img.shields.io/pypi/l/dualpy)](https://github.com/hmackk/dualpy/blob/main/LICENSE)

**Lightweight forward-mode automatic differentiation for NumPy.**

Dualpy lets you compute exact derivatives of ordinary NumPy code. It hooks
into NumPy's own dispatch protocols (`__array_ufunc__` and
`__array_function__`) to propagate dual numbers alongside the primal
computation, so your existing functions work unchanged. NumPy is the only
dependency — no compilation step, no new array API to learn, and nothing
heavy to install.

## At a glance

```python
import numpy as np
from dualpy import gradient

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

grad = gradient(rosenbrock)
grad(np.array([1.0, 1.0]))  # array([0., 0.])  — at the minimum
```

## What's covered

Dualpy ships derivative rules for a broad set of NumPy operations — elementary
arithmetic, trigonometry, exponentials, linear algebra (including matrix
decompositions like SVD, QR, Cholesky, and eigendecomposition), reductions,
shape manipulation, and numerical routines — covering the needs of scientific
research, engineering, and education.

## Features

- **Complex derivatives.** Complex-valued arrays are supported and produce
  correct derivatives for
  [holomorphic functions](https://en.wikipedia.org/wiki/Holomorphic_function).
- **Higher-order derivatives.** Exact \(n\)-th order derivatives via nested
  forward-mode (forward-over-forward).
- **Multi-argument differentiation.** Differentiate with respect to any
  positional argument (or several) using `argnums`.

## Limitations

- **Forward-mode only.** Most efficient when the number of inputs is small
  relative to the number of outputs. For high-dimensional inputs with scalar
  output (e.g. neural network loss), reverse-mode (backpropagation) is
  faster — consider JAX or PyTorch for that use case.
- **Registered operations only.** If your code calls a NumPy function that
  dualpy hasn't registered, you'll get `NotImplementedError`. The set of
  registered operations covers common scientific computing needs, and it will
  be expanded further.
- **No GPU support.** Dualpy operates on CPU NumPy arrays.
