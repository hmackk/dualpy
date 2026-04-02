# Getting Started

## Installation

```bash
pip install dualpy
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add dualpy
```

## How it works

Dualpy uses **dual numbers** to compute exact derivatives in a single forward
pass. A dual number extends a real number with an infinitesimal part
\(\varepsilon\) (where \(\varepsilon^2 = 0\)):

\[
f(a + \varepsilon) = f(a) + f'(a)\,\varepsilon
\]

By evaluating your function with dual-number inputs, the derivative
\(f'(a)\) is carried alongside the primal value \(f(a)\) automatically:
no symbolic manipulation, no finite differences, no backward pass. This is
called **forward-mode automatic differentiation**.

Dualpy implements this by hooking into NumPy's `__array_ufunc__` and
`__array_function__` protocols, so every supported NumPy operation
propagates both the primal and tangent (derivative) through the
computation graph transparently.

## Which function should I use?

| Your function signature | Use | Example |
|---|---|---|
| \(f: \mathbb{R} \to \mathbb{R}\) | `derivative` | `derivative(np.sin)(0.0)` |
| \(f: \mathbb{R}^n \to \mathbb{R}\) | `gradient` | `gradient(loss)(params)` |
| \(f: \mathbb{R}^n \to \mathbb{R}^m\) | `jacobian` | `jacobian(f)(x)` |
| \(f: \mathbb{R}^n \to \mathbb{R}\), need 2nd derivatives | `hessian` or `hvp` | `hessian(loss)(params)` |
| \(n\)-th derivative of \(f: \mathbb{R} \to \mathbb{R}\) | `nth_derivative` | `nth_derivative(f, 3)(x)` |
| \(f: \mathbb{R}^n \to \mathbb{R}\), directional gradient | `gradient(f, v=direction)` | `gradient(f, v=v)(x)` |

For the underlying primitive that all of the above build on, see
[`jvp`](api/differentiation.md#dualpy.jvp).

## Quickstart

### Gradient

Compute the gradient of a scalar-valued function:

```python
import numpy as np
from dualpy import gradient

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

grad = gradient(rosenbrock)
grad(np.array([1.0, 1.0]))  # array([0., 0.]), at the minimum
```

### Jacobian

Compute the full Jacobian matrix of a vector-valued function:

```python
from dualpy import jacobian

def f(x):
    return np.stack([x[0]**2, x[0] * x[1]])

jacobian(f)(np.array([2.0, 3.0]))
# array([[4., 0.],
#        [3., 2.]])
```

### Derivative

Compute the derivative of a scalar-to-scalar function:

```python
from dualpy import derivative

df = derivative(np.sin)
df(0.0)  # cos(0) = 1.0
```

### Hessian

Compute the Hessian matrix via forward-over-forward:

```python
from dualpy import hessian

def quadratic(x):
    return x[0]**2 + 3 * x[1]**2

hessian(quadratic)(np.array([1.0, 1.0]))
# array([[2., 0.],
#        [0., 6.]])
```

### Hessian-vector product

Compute \(H(x) \cdot v\) in O(n) without forming the full Hessian:

```python
from dualpy import hvp

f = lambda x: x[0]**2 + 3 * x[1]**2
hvp(f, np.array([1.0, 0.0]))(np.array([1.0, 1.0]))
# array([2., 0.]), equivalent to hessian(f)(x) @ v, but O(n)
```

### Low-level JVP

The Jacobian-vector product is the fundamental primitive that all higher-level
functions build on:

```python
from dualpy import jvp

f = lambda x: np.sin(x) * np.exp(x)
primal, tangent = jvp(f, np.array(1.0), np.array(1.0))
# tangent is df/dx at x=1
```

### Higher-order derivatives

Compute exact \(n\)-th order derivatives via nested forward-mode:

```python
from dualpy import nth_derivative

f = lambda x: x ** 4
nth_derivative(f, 3)(1.0)  # f'''(x) = 24x, so f'''(1) = 24.0
```

### Multi-argument differentiation

Use `argnums` to differentiate with respect to any positional argument:

```python
from dualpy import gradient

f = lambda x, y: np.sum(x**2) + y**2
gradient(f, argnums=0)(np.array([3.0, 4.0]), 1.0)  # array([6., 8.])

# Differentiate with respect to multiple arguments at once
from dualpy import derivative

g = lambda x, y: x**2 * y
derivative(g, argnums=(0, 1))(3.0, 2.0)  # (12.0, 9.0)
```

### Complex derivatives

Complex-valued arrays are supported and produce correct derivatives for
holomorphic functions:

```python
f = lambda z: z ** 2
derivative(f)(1.0 + 2.0j)  # 2 + 4j, since d/dz(z²) = 2z
```

### Vector calculus

Dualpy includes differential operators for vector fields:

```python
from dualpy import curl, divergence, laplacian

# Divergence of the identity field
F = lambda x: x
divergence(F)(np.array([1.0, 2.0, 3.0]))  # 3.0

# Laplacian of a quadratic
g = lambda x: x[0]**2 + x[1]**2 + x[2]**2
laplacian(g)(np.array([1.0, 2.0, 3.0]))  # 6.0

# Curl of a rotation field
R = lambda x: np.stack([x[1], -x[0], np.zeros_like(x[0])])
curl(R)(np.array([1.0, 2.0, 3.0]))  # array([ 0.,  0., -2.])
```

## Common pitfalls

!!! warning "Use `np.stack`, not `np.array`, for vector outputs"

    When building multi-element outputs inside a function you want to
    differentiate, always use `np.stack([...])` rather than `np.array([...])`.

    Due to a limitation in NumPy's dispatch mechanism, `np.array` does not
    correctly propagate derivative information through lists of intermediate
    results. This causes dualpy to fall back to a slower element-by-element
    recovery path and emit a `UserWarning`.

    ```python
    # ✗ Avoid
    def f(x):
        return np.array([x[0]**2, x[1]**2])

    # ✓ Prefer
    def f(x):
        return np.stack([x[0]**2, x[1]**2])
    ```

## Next steps

- [API Reference](api/differentiation.md): full signatures and docstrings
  for every public function.
- [Supported Operations](supported-operations.md): every NumPy operation
  dualpy can differentiate through.
