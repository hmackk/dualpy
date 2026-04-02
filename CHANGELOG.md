# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.1.0] - 2026-04-02

### Added

- `DualArray` type with NumPy protocol dispatch (`__array_ufunc__`, `__array_function__`).
- Forward-mode AD API: `jvp`, `jacobian`, `derivative`, `nth_derivative`, `gradient`, `hessian`, `hvp`.
- Vector calculus operators: `curl`, `divergence`, `laplacian`.
- 50+ ufunc derivative rules: elementary arithmetic, trigonometric, hyperbolic, exponential/logarithmic, comparison, logical, and rounding operations.
- 60+ array routine rules: linear algebra (including `det`, `inv`, `solve`, `cholesky`, `eigh`, `svd`, `qr`, `lstsq`), reductions, shape manipulation, construction, searching, and numerical routines.
- Nested forward-mode for exact higher-order derivatives.
- Multi-argument differentiation via `argnums`.

[Unreleased]: https://github.com/hmackk/dualpy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hmackk/dualpy/releases/tag/v0.1.0
