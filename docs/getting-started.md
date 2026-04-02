# Getting Started

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

Compute the gradient of a scalar-valued function:

```python
import numpy as np
from dualpy import gradient

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

grad = gradient(rosenbrock)
grad(np.array([1.0, 1.0]))  # array([0., 0.])  — at the minimum
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
nth_derivative(f, 3)(1.0)  # f'''(x) = 24x, at x=1 → 24.0
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
