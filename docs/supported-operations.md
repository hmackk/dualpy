# Supported Operations

Dualpy registers derivative rules for the NumPy operations listed below.
Any operation **not** on this list will raise `NotImplementedError` when
called on dual-number inputs. Coverage is expanded in each release.

## Ufuncs

### Arithmetic

| NumPy ufunc | Notes |
|---|---|
| `add` | |
| `subtract` | |
| `multiply` | |
| `divide` / `true_divide` | |
| `power` | |
| `float_power` | |
| `negative` | |
| `positive` | |
| `absolute` / `abs` | |
| `fabs` | |
| `square` | |
| `reciprocal` | |
| `cbrt` | |
| `fmod` | |
| `remainder` / `mod` | |
| `copysign` | |
| `matmul` | |
| `maximum` | |
| `minimum` | |

### Trigonometric

| NumPy ufunc | Notes |
|---|---|
| `sin` | |
| `cos` | |
| `tan` | |
| `arcsin` | |
| `arccos` | |
| `arctan` | |
| `arctan2` | |
| `hypot` | |
| `degrees` | |
| `radians` | |
| `deg2rad` | |
| `rad2deg` | |

### Hyperbolic

| NumPy ufunc | Notes |
|---|---|
| `sinh` | |
| `cosh` | |
| `tanh` | |
| `arcsinh` | |
| `arccosh` | |
| `arctanh` | |

### Exponential & logarithmic

| NumPy ufunc | Notes |
|---|---|
| `exp` | |
| `exp2` | |
| `expm1` | |
| `log` | |
| `log2` | |
| `log10` | |
| `log1p` | |
| `logaddexp` | |
| `logaddexp2` | |
| `sqrt` | |

### Comparison

| NumPy ufunc | Notes |
|---|---|
| `equal` | Returns plain array (no tangent) |
| `not_equal` | Returns plain array |
| `greater` | Returns plain array |
| `greater_equal` | Returns plain array |
| `less` | Returns plain array |
| `less_equal` | Returns plain array |

### Logical & query

| NumPy ufunc | Notes |
|---|---|
| `logical_and` | Returns plain array |
| `logical_or` | Returns plain array |
| `logical_xor` | Returns plain array |
| `logical_not` | Returns plain array |
| `isnan` | Returns plain array |
| `isinf` | Returns plain array |
| `isfinite` | Returns plain array |
| `signbit` | Returns plain array |

### Rounding

| NumPy ufunc | Notes |
|---|---|
| `sign` | Zero tangent (piecewise constant) |
| `heaviside` | Zero tangent |
| `floor` | Zero tangent |
| `ceil` | Zero tangent |
| `trunc` | Zero tangent |
| `rint` | Zero tangent |

---

## Array routines

### Linear algebra

| NumPy function | Notes |
|---|---|
| `np.dot` | |
| `np.inner` | |
| `np.outer` | |
| `np.tensordot` | |
| `np.einsum` | Arbitrary number of operands |
| `np.trace` | |
| `np.cross` | |
| `np.vdot` | |
| `np.kron` | |
| `np.linalg.norm` | Supports `ord`, `axis`, `keepdims` |
| `np.linalg.det` | |
| `np.linalg.inv` | |
| `np.linalg.solve` | Both A and b may be dual |
| `np.linalg.cholesky` | |
| `np.linalg.eigh` | Symmetric/Hermitian eigendecomposition |
| `np.linalg.svd` | Full and reduced modes |
| `np.linalg.qr` | |
| `np.linalg.lstsq` | |
| `np.linalg.matrix_power` | Positive, negative, and zero exponents |
| `np.linalg.multi_dot` | |

### Reductions

| NumPy function | Notes |
|---|---|
| `np.sum` | |
| `np.mean` | |
| `np.prod` | |
| `np.max` | |
| `np.min` | |
| `np.var` | Supports `ddof` |
| `np.std` | Supports `ddof` |
| `np.cumsum` | |
| `np.cumprod` | |
| `np.argmax` | Returns plain integer (no tangent) |
| `np.argmin` | Returns plain integer |
| `np.nansum` | |
| `np.nanmean` | |
| `np.nanvar` | |
| `np.nanstd` | |
| `np.nanmax` | |
| `np.nanmin` | |
| `np.average` | Supports `weights` |
| `np.median` | |
| `np.all` | Returns plain array |
| `np.any` | Returns plain array |
| `np.count_nonzero` | Returns plain integer |

### Shape manipulation

| NumPy function | Notes |
|---|---|
| `np.concatenate` | |
| `np.stack` | |
| `np.hstack` | |
| `np.vstack` | |
| `np.dstack` | |
| `np.column_stack` | |
| `np.reshape` | |
| `np.transpose` | |
| `np.swapaxes` | |
| `np.moveaxis` | |
| `np.squeeze` | |
| `np.expand_dims` | |
| `np.ravel` | |
| `np.split` | |
| `np.array_split` | |
| `np.tile` | |
| `np.repeat` | |
| `np.flip` | |
| `np.fliplr` | |
| `np.flipud` | |
| `np.roll` | |
| `np.broadcast_to` | |
| `np.atleast_1d` | |
| `np.atleast_2d` | |
| `np.atleast_3d` | |
| `np.pad` | Tangent padded with zeros |
| `np.rot90` | |
| `np.take` | |
| `np.take_along_axis` | |
| `np.insert` | |
| `np.delete` | |
| `np.append` | |
| `np.select` | |

### Construction

| NumPy function | Notes |
|---|---|
| `np.array` | From list of dual arrays |
| `np.zeros_like` | |
| `np.ones_like` | |
| `np.full_like` | |
| `np.empty_like` | Returns zeros (safe default) |
| `np.copy` | |
| `np.linspace` | |
| `np.arange` | |
| `np.eye` | |
| `np.identity` | |
| `np.diag` | |
| `np.diagonal` | |
| `np.meshgrid` | |
| `np.triu` | |
| `np.tril` | |

### Searching & sorting

| NumPy function | Notes |
|---|---|
| `np.where` | |
| `np.clip` | |
| `np.sort` | |
| `np.argsort` | Returns plain integer array |
| `np.searchsorted` | Returns plain integer array |
| `np.nonzero` | Returns plain index arrays |
| `np.flatnonzero` | Returns plain index array |
| `np.extract` | |

### Numerical

| NumPy function | Notes |
|---|---|
| `np.diff` | |
| `np.convolve` | |
| `np.correlate` | |
| `np.interp` | |
| `np.trapezoid` | Supports `x` and `dx` |
| `np.sinc` | |
| `np.gradient` | Numerical gradient (not to be confused with `dualpy.gradient`) |
