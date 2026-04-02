from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np
import numpy.typing as npt

from .registry import FUNC_REGISTRY, UFUNC_REGISTRY


class DualArray:
    """A dual number array for forward-mode automatic differentiation.

    Pairs a *primal* value (the ordinary computation) with a *tangent*
    value (the directional derivative) and propagates both through NumPy
    operations via ``__array_ufunc__`` and ``__array_function__``.

    Parameters
    ----------
    primal : ndarray or DualArray
        The value of the computation.  When a ``DualArray`` is passed,
        nested dual numbers are created for higher-order derivatives.
    tangent : ndarray, DualArray, or None
        The derivative seed.  Must be broadcast-compatible with *primal*.
        Defaults to zeros (i.e. a constant with no derivative contribution).
    """

    primal: npt.NDArray | DualArray
    tangent: npt.NDArray | DualArray

    def __init__(
        self,
        primal: npt.NDArray | DualArray,
        tangent: npt.NDArray | DualArray | None = None,
    ) -> None:
        self.primal = primal
        if tangent is None:
            if isinstance(primal, DualArray):
                self.tangent = DualArray(
                    np.zeros_like(primal.primal),
                    np.zeros_like(primal.tangent),
                )
            else:
                self.tangent = np.zeros_like(primal)
        else:
            if tangent.ndim > primal.ndim:
                # Tangent may carry extra trailing dimensions for batch
                # differentiation (e.g. Jacobian columns). Accept if
                # primal shape is a prefix of tangent shape.
                if tangent.shape[: primal.ndim] != primal.shape and primal.ndim > 0:
                    raise ValueError(
                        f"tangent shape {tangent.shape} is not compatible "
                        f"with primal shape {primal.shape}"
                    )
            else:
                _ = np.broadcast_shapes(primal.shape, tangent.shape)
            self.tangent = tangent

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> DualArray | Any:
        if method != "__call__":
            return NotImplemented
        if ufunc.__name__ in UFUNC_REGISTRY:
            inputs_ = []
            for x in inputs:
                if isinstance(x, DualArray):
                    inputs_.append((x.primal, x.tangent))
                elif isinstance(x, np.ndarray):
                    inputs_.append((x, np.zeros_like(x)))
                elif isinstance(x, np.generic):
                    arr = np.asarray(x)
                    inputs_.append((arr, np.zeros_like(arr)))
                else:
                    inputs_.append(x)
            result = UFUNC_REGISTRY[ufunc.__name__](*inputs_, **kwargs)
            if isinstance(result, tuple | list):
                out_primal, out_tangent = result
                return DualArray(out_primal, out_tangent)
            return result
        else:
            raise NotImplementedError(f"ufunc {ufunc.__name__} not implemented")

    def __array_function__(
        self,
        func: Any,
        types: Collection[type],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> DualArray | Any:
        if func.__name__ not in FUNC_REGISTRY:
            raise NotImplementedError(f"array function {func.__name__} not implemented")
        if not all(issubclass(t, DualArray | np.ndarray) for t in types):
            return NotImplemented
        result = FUNC_REGISTRY[func.__name__](*args, **kwargs)
        if isinstance(result, tuple):
            out_primal, out_tangent = result
            return DualArray(out_primal, out_tangent)
        return result

    def __repr__(self) -> str:
        return f"DualArray({self.primal}, {self.tangent})"

    # Indexing

    def __getitem__(self, key: Any) -> DualArray:
        return DualArray(self.primal[key], self.tangent[key])

    def __setitem__(self, key: Any, value: DualArray | npt.ArrayLike) -> None:
        if not isinstance(value, DualArray):
            value = DualArray(np.asarray(value))
        self.primal[key] = value.primal
        self.tangent[key] = value.tangent

    def __delitem__(self, key: Any) -> None:
        self.primal = np.delete(self.primal, key)
        self.tangent = np.delete(self.tangent, key)

    def __len__(self) -> int:
        return len(self.primal)

    # Properties

    @property
    def T(self) -> DualArray:
        return DualArray(self.primal.T, self.tangent.T)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.primal.shape

    @property
    def dtype(self) -> np.dtype:
        return self.primal.dtype

    @property
    def size(self) -> int:
        return self.primal.size

    @property
    def ndim(self) -> int:
        return self.primal.ndim

    # Unary operators

    def __neg__(self) -> DualArray:
        return DualArray(-self.primal, -self.tangent)

    def __abs__(self) -> DualArray:
        return DualArray(np.abs(self.primal), np.sign(self.primal) * self.tangent)

    # Arithmetic (forward)

    def _wrap(self, other: DualArray | npt.ArrayLike) -> DualArray:
        if isinstance(other, DualArray):
            return other
        return DualArray(np.asarray(other))

    def __add__(self, other: DualArray | npt.ArrayLike) -> DualArray:
        return np.add(self, self._wrap(other))  # ty:ignore[no-matching-overload]

    def __sub__(self, other: DualArray | npt.ArrayLike) -> DualArray:
        return np.subtract(self, self._wrap(other))  # ty:ignore[no-matching-overload]

    def __mul__(self, other: DualArray | npt.ArrayLike) -> DualArray:
        return np.multiply(self, self._wrap(other))  # ty:ignore[no-matching-overload]

    def __truediv__(self, other: DualArray | npt.ArrayLike) -> DualArray:
        return np.divide(self, self._wrap(other))  # ty:ignore[no-matching-overload]

    def __pow__(self, n: DualArray | npt.ArrayLike) -> DualArray:
        return np.power(self, n)  # ty:ignore[no-matching-overload]

    def __matmul__(self, other: DualArray | npt.ArrayLike) -> DualArray:
        return np.matmul(self, self._wrap(other))  # ty:ignore[no-matching-overload]

    # Arithmetic (reverse)

    def __radd__(self, other: DualArray | npt.ArrayLike) -> DualArray:
        return np.add(self._wrap(other), self)  # ty:ignore[no-matching-overload]

    def __rsub__(self, other: DualArray | npt.ArrayLike) -> DualArray:
        return np.subtract(self._wrap(other), self)  # ty:ignore[no-matching-overload]

    def __rmul__(self, other: DualArray | npt.ArrayLike) -> DualArray:
        return np.multiply(self._wrap(other), self)  # ty:ignore[no-matching-overload]

    def __rtruediv__(self, other: DualArray | npt.ArrayLike) -> DualArray:
        return np.divide(self._wrap(other), self)  # ty:ignore[no-matching-overload]

    def __rpow__(self, other: DualArray | npt.ArrayLike) -> DualArray:
        return np.power(self._wrap(other), self)  # ty:ignore[no-matching-overload]

    def __rmatmul__(self, other: DualArray | npt.ArrayLike) -> DualArray:
        return np.matmul(self._wrap(other), self)  # ty:ignore[no-matching-overload]

    # Comparison (return plain arrays, not DualArray)

    def __eq__(self, other: object) -> Any:
        return np.equal(self, self._wrap(other))

    def __ne__(self, other: object) -> Any:
        return np.not_equal(self, self._wrap(other))

    def __gt__(self, other: DualArray | npt.ArrayLike) -> Any:
        return np.greater(self, self._wrap(other))

    def __ge__(self, other: DualArray | npt.ArrayLike) -> Any:
        return np.greater_equal(self, self._wrap(other))

    def __lt__(self, other: DualArray | npt.ArrayLike) -> Any:
        return np.less(self, self._wrap(other))

    def __le__(self, other: DualArray | npt.ArrayLike) -> Any:
        return np.less_equal(self, self._wrap(other))
